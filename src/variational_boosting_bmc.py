# -*- coding: utf-8 -*-
import math
import functools

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np

from . import utils
from .bmc import BMC_FM,BMC_QM
from .bmc_sparse import BMC_SFM,BMC_SQM

DEBUG_MODE = True


def in_and_where(tensor1d,tensor2d):
    if tensor2d.numel() == 0:
        return False,np.array([])
    else:
        inside = (tensor1d == tensor2d).prod(dim=1)
        inside_ind = torch.tensor(np.argwhere(inside.numpy()).flatten())
        return bool(sum(inside_ind)),inside_ind
        

class VariationalBoosting(object):
    """
        Variational boosting with BMC surrogate.
    """
    def __init__(self,dim,logjoint,samples,mu0,cov0,
                 bmc_type="FM",normalization_mode="normalize",
                 training_space="gspace",
                 constraints = None,
                 numstab=-math.inf,denstab=-20.0,refstab=None,
                 evaluations=None,device=None,
                 **kwargs):
        """
            dim : dimension of the space
            logjoint : a torch function of theta
            samples : initial samples
            mu0 : initial mean
            cov0 : initial cov
            bmc_type : type of the BMC. Options : "FM","QM","SFM","SQM". Default: "FM"
            normalization_mode : mode of normalization for BC. 
                    Options : "zeromax","normalize","none" or None.
                    Default : "normalize"
            training_space : training space for GP. Options : "gspace","fspace"
                             Default : "gspace"
            constraint : constraint for the logjoint function. Default: None
            numstab : stabilizer constant for numerator. Default: -math.inf
            denstab : stabilizer constant for denominator. Default: -math.inf
            refstab : reference stabilization constant. If None, 
                      is chosen to the maximum of initial evaluations.
                      Default: None
            evaluations : evaluations (can be None). Default: None
            device : device (can be None). Default: None
            theta0 : initial theta for BMC. Default: 1.0
            l0 : initial lengthscale for theta. Default : 1.0
            noise : BMC noise. Default : 1e-4
            fixed_noise : whether to keep noise fixed in BMC. Default: True
            ninducing : Number of inducing points. Only for SPARSE BMC. Default : 50
            PUT THE REST OF PARAMETERS
        """
        self.dim = dim
        self.logjoint = logjoint
        self.constraints = constraints
        self.numstab = numstab
        self.denstab = denstab
        self.refstab = refstab
        self.bmc_type = bmc_type
        self.normalization_mode = normalization_mode
        self.training_space = training_space
        if self.training_space == "fspace" and self.normalization_mode != "zeromax":
            print("fspace can only be used with zeromax normalization. normalization mode changed")
            self.normalization_mode = "zeromax"
        if device == None:
            self.device = samples.device.type
        self.initialize(mu0,cov0,samples,evaluations,self.device,**kwargs)
            
    def current_logq(self,theta,stabilizer=-math.inf,flatten=False):
        """
            Arguments:
                theta : and (nsamples,ndim) tensor or a (ndim,) tensor
                stabilizer : whether to use a stabilizer. DEPRECATED
                flatten : whether to flatten result. For internal use only
            Returns:
                evaluations of current log density of proposal at theta
        """
        if stabilizer==-math.inf and flatten==False:
            return self.distribution.logprob(theta)
        if theta.dim() == 1:
            theta = theta.reshape(1,-1)
        #theta : (*,D)
        logqs = utils.logmvnbatch(theta,self.mu_t,self.cov_t) #(*,Q)
        if stabilizer == -math.inf:
            weights = self.weights #(Q,)
        else:
            logqs = torch.cat([logqs,stabilizer*torch.ones(*theta.shape[:-1],1,
                                                           device=logqs.device)],dim=1) #(*,Q+1)
            weights = torch.cat([self.weights,torch.ones(1,device=logqs.device)]) #(Q+1,)
        res = utils.logexpsum(logqs,weights,flatten) #(*,1)
        return res
    
    def current_q(self,theta,flatten=False):
        """
            Arguments:
                theta : and (nsamples,ndim) tensor or a (ndim,) tensor
                flatten : whether to flatten result. For internal use only
            Returns:
                evaluations of current density of proposal at theta
        """
        if theta.dim() == 1:
            theta = theta.reshape(1,-1)
        res_ = utils.logmvnbatch(theta,self.mu_t,self.cov_t)
        res_ = torch.exp(res_)
        res = (self.weights*res_).sum(dim=-1,keepdim=not flatten)
        return res

    def cutweights(self,cutoff=1e-6):
        """
            Eliminate every component with weight < cutoff
            cutoff : cutoff point. Default : 1e-6
        """
        remainers = ~(self.weights<cutoff)
        self.weights = self.weights[remainers]
        self.weights = self.weights/sum(self.weights)
        self.mu_t = self.mu_t[remainers,:]
        self.cov_t = self.cov_t[remainers,:]
        self.num_mixtures = len(self.weights)
    
    def elbo_var(self,approximate=False):
        if not approximate:
            _,var = self.bmcmodel.evaluate_integral_mixture(self.mu_t,self.cov_t,self.weights,
                                                            retvar=True,diag=True)
        else:
            _,var = self.bmcmodel.evaluate_integral(self.currentq_mean,self.currentq_cov,
                                                    retvar=True,diag=False)
        return var
    
    def evaluate_logjoint(self,theta,flatten=False,stabilize=True):
        """
            Arguments:
                theta : an (nsamples,ndim) tensor or a (ndim,) tensor
                stabilizer : whether to use a stabilizer. For internal use only
                flatten : whether to flatten result. For internal use only
            Returns:
                evaluations of unnormalized true density at theta
        """
        if theta.dim() == 1:
            if self.constraints and not self.constraints(theta):
                res = self.num_stabilizer*torch.ones(1,device=theta.device)
            else:
                res = self.logjoint(theta)
            if (res < self.num_stabilizer and stabilize) or torch.isnan(res):
                res = self.num_stabilizer*torch.ones_like(res)
            return res
        else:
            result = torch.zeros(theta.shape[0],1,device=theta.device)
            for i in range(theta.shape[0]):
                result[i] = self.evaluate_logjoint(theta[i,:],stabilize=stabilize)
            return result

    def evidence_lower_bound(self,nsamples=1000,calc_type="simple"):
        """
            Return ELBO estimate of current variational distribution and 
            unnormalized true density
            Arguments:
                nsamples : number of samples to be used in estimation
                calc_type : "simple", if there is no need to stochastic gradient,
                            "complex" if there is need of stochastic gradient.
                            Usually for internal use only
        """
        normed_logjoint = self.bmcmodel.evaluate_integral_mixture(self.mu_t,
                                                           self.cov_t,
                                                           self.weights,
                                                           diag=True)
        logjoint = self.evals_std*normed_logjoint + self.evals_mean
        if calc_type == "simple":
            samples = self._sample_from_current(nsamples)
            entropy = -self.current_logq(samples).mean()
        else: #Suitable for backprop on weights
            samples_ = torch.randn(nsamples,self.num_mixtures,self.dim,
                                   device=self.device) #(N,Q,D)
            samples_ = torch.sqrt(self.cov_t)*samples_ + self.mu_t #(N,Q,D)
            entropy_ = -self.current_logq(samples_,flatten=True) #(N,Q)
            entropy_ = entropy_.mean(dim=0) #(Q,)
            entropy = (entropy_*self.weights).sum() #(,)
        return logjoint + entropy

    def initialize(self,mu0,cov0,samples,evaluations=None,device="cpu",**kwargs):
        """
           Initialize the model. Usually for internal use only (although not necessarily) 
        """
        self._initialize_vb(mu0,cov0,device,**kwargs)
        evals_norm = self._initialize_with_samples(samples,evaluations,**kwargs)
        self._initialize_bmc(samples,evals_norm,device,**kwargs)
    
    def kullback_proposal_bmc(self,nsamples):
        """
            Tries to estimate the KL divergence between the variational 
            proposal and the GP. May be useful for tracking convergence 
            (although use with caution)
        """
        term1 = -self.evidence_lower_bound(nsamples)
        #Term 2 is the actual evidence IS estimate
        samples = self._sample_from_current(nsamples)
        qs = self.current_q(samples)
        logpgp = self.scaled_prediction(samples)
        pgp = torch.exp(logpgp)
        term2 = torch.log((pgp/qs).mean())
        return(term1 + term2)

        
    def optimize_bmc_model(self,lr=0.5,maxiter=50,**kwargs):
        """
            Optimize hyperparameters of the BMC, or GP surrogate model. Uses Adam
            Arguments:
                lr : learning rate. Default : 0.5
                maxiter : number of training iterations. Default : 50
                verbose : level of verbosity
        """
        self.bmcmodel.optimize_model(lr=lr,
                                     training_iter=maxiter,
                                     training_space=self.training_space,
                                     **kwargs)
        self.bmcmodel.make_weights_vector()
        
    def update(self,**kwargs):
        """
            Seek a new component by boosting. Inner optimizers are all Adam.
            Arguments:
                verbose : level of verbosity. Default: 1
                maxiter_nc : number of iterations of new component choosing optimization. Default: 300
                lr_nc : learning rate of new component choosing optimization. Default: 0.1
                distrib_type_nc : distribution of initial variance sampling for new component. Default: "HN"
                nsamples_nc : number of samples for estimating cross-entropy in RELBO. Default: 100
                b_sn : learning_constant for new weight determination optimization. Default: 0.1
                n_samples_sn : number of samples to estimate cross-entropy in new weight optimization. Default 100
                n_iter_sn : number of optimization iterations. Default: 100
                n_iter_min : minimum number of iterations before checking for convergence. Default: 100
                alpha_tol : tolerance. Default: 1e-4
        """
        verbose = kwargs.get("verbose",1)
        if verbose : print("calculating mu and cov...")
        mu_new,cov_new = self._find_new_components(**kwargs)
        if verbose : print("calculating alpha...")
        alpha_new = self._find_alpha_new(mu_new,cov_new,**kwargs)
        self.mu_t = torch.cat([self.mu_t,mu_new.reshape(1,-1)],dim=0)
        self.cov_t = torch.cat([self.cov_t,cov_new.reshape(1,-1)],dim=0)
        self.weights = torch.cat([(1.0-alpha_new)*self.weights,
                                  alpha_new*torch.ones(1,device=self.weights.device)])
        self.num_mixtures += 1
        if verbose >= 1 : print("weights:",self.weights)
    
    def update_bmcmodel(self,acquisition,num_evals=100,
                        mode="optimizing",
                        lr=0.1,acq_reg=1e-4,verbose=1,
                        **kwargs):
        """
            Update the BMC (or surrogate GP) model
            Only implemented for dense models for now
            Arguments:
                acquisition : acquisition function. Options are
                    "prospective","uncertainty_sampling",
                    "mmlt","mmlt_prospective"
                lr : learning rate for optimization
                num_evals : number of training iterations for optimization,
                            or sampling points for choosing
                verbose : level of verbosityf
        """
        vreg = kwargs.get("vreg",None)
        if vreg: acq_reg = vreg
        if self.bmc_type in ["SCM","SQM","SFM"]:
            evals_new,prediction_new = self._update_sparse_model(acquisition,num_evals,mode,
                                                                 lr,verbose,acq_reg)
        else:
            evals_new,prediction_new = self._update_dense_model(acquisition,num_evals,mode,
                                                                lr,verbose,acq_reg)
        return evals_new,prediction_new
    
    def update_full(self,nsamples=250,lr=0.1,maxiter=300,verbose=1,
                         cutoff=1e-6):
        """
            Update all variational proposal parameters.
            Arguments:
                nsamples : get number of samples for entropy. Default : 250
                lr : learning rate for optimization. Default : 0.1
                maxiter : number of training iterations for optimization. Default : 300
                verbose : level of verbosity. Default : 1
                cutoff : tolerance for cutting weights. Default : 1e-6
        """
        weights_backup = self.weights.detach().clone()
        mu_backup = self.mu_t.detach().clone()
        cov_backup = self.cov_t.detach().clone()
        raw_weights = utils.invsoftplus(self.weights.detach(),minvalue=0.0)
        raw_mu = self.mu_t.detach().clone()
        raw_cov = utils.invsoftplus(self.cov_t.detach().clone(),minvalue=0.0)
        optimizer = torch.optim.Adam([raw_weights,raw_mu,raw_cov],lr=lr)
        for i in range(maxiter):
            optimizer.zero_grad()
            raw_weights.requires_grad = True
            raw_mu.requires_grad = True
            raw_cov.requires_grad = True
            weights = utils.softplus(raw_weights,minvalue=0.0)
            weights = weights/torch.sum(weights)
            self.weights = weights
            self.mu_t = raw_mu
            self.cov_t = utils.softplus(raw_cov)
            self.cutweights(cutoff)
            loss = -self.evidence_lower_bound(nsamples,calc_type="complex")
            if verbose >= 2:
                print(self.weights)
                print(self.mu_t)
                print(self.cov_t)
                print(loss)
                print(i)
            if torch.any(torch.isnan(loss) | torch.isinf(loss)):
                print("Problem with optimization, reverting")
                self.weights = weights_backup
                self.mu_t = mu_backup
                self.cov_t = cov_backup
                return
            else:
                loss.backward()
                optimizer.step()
        self.weights = self.weights.detach().clone()
        self.mu_t = self.mu_t.detach()
        self.cov_t = self.cov_t.detach()
        self.cutweights(cutoff)
        if verbose >= 1:
            print(loss)
            print(self.weights)
        return

    def samples_q(self,nsamples=1):
        """
            Sample from current variational proposal.
            Arguments : 
                nsamples: number of samples
            Returns:
                (nsamples,dim) tensor of samples
        """
        return self._sample_from_current(nsamples=nsamples,flatten=False)

    def samples_proposal(self,nsamples=1):
        """
            Deprecated. Use samples_q.
        """
        return self._sample_from_current(nsamples=nsamples,flatten=False)
    
    def save_distribution(self,filename):
        """
            Save variational distribution as a MixtureNormalDiagCov class,
            consisting of:
                attributes:
                    mu_t : (N,D) tensor
                    cov_t : (N,D) tensor
                    weights : (D,) tensor
                properties
                    num_mixtures : number of mixtures
                    dim : dimension
                    cov2_t : cov_t in (N,N,D) format
                methods:
                    mean(self) : return (D,) mean of distribution
                    cov(self) : return (D,D) covariance of distribution
                    sample(self,nsamples,flatten=False) : return samples of distribution
                    logprob(self,theta) : logprob of theta
        """
        torch.save(self.distribution,filename)

    def scaled_prediction(self,x):
        return self.evals_std*self.bmcmodel.prediction(x,cov="none") + self.evals_mean

    @property
    def mu_t(self):
        return self.distribution.mu_t
    
    @mu_t.setter
    def mu_t(self,mu):
        self.distribution.mu_t = mu
    
    @property
    def cov_t(self):
        return self.distribution.cov_t
    
    @cov_t.setter
    def cov_t(self,cov):
        self.distribution.cov_t = cov
    
    @property
    def weights(self):
        return self.distribution.weights
    
    @weights.setter
    def weights(self,w):
        self.distribution.weights = w
    
    @property
    def cov2_t(self):
        return torch.diag_embed(self.cov_t)
    
    @property
    def samples(self):
        return self.bmcmodel.samples
    
    @property
    def evaluations(self):
        return self.evals_std*self.evals_norm + self.evals_mean
    
    @property
    def evals_norm(self):
        return self.bmcmodel.evaluations
    
    @property
    def currentq_mean(self):
        return self.distribution.mean()
    
    @property
    def currentq_cov(self):
        return self.distribution.cov()
    
    @property
    def num_stabilizer(self):
        if self.numstab == -math.inf:
            return self.numstab
        if self.refstab != None:
            ref = self.refstab
        else:
            ref = 0.0
#            ref = torch.max(self.evaluations)
        return self.numstab + ref
    
    @property
    def den_stabilizer(self):
        if self.denstab == -math.inf:
            return self.denstab
        if self.refstab != None:
            ref = self.refstab
        else:
            ref = 0.0
#            ref = torch.max(self.evaluations)
        return self.denstab + ref

    def _evaluate_loss(self,theta0,nsamples=100,verbose=0):
        #Maximize RELBO (loss is negative relbo)
        mu = theta0[:self.dim]
        raw_cov = theta0[self.dim:]
        cov = utils.softplus(raw_cov)
#        lambd = 1.0/math.sqrt(1self.num_mixtures+1)
        lambd = torch.rand(1)
        #self-entropy term
        self_entropy_term = 0.5*self.dim*math.log(2*math.pi*math.e) + \
                            0.5*torch.sum(torch.log(cov))
        #logjoint term
        normed_logjoint_term = self.bmcmodel.evaluate_integral(mu,cov,diag=True)
        logjoint_term = self.evals_std*normed_logjoint_term + self.evals_mean
        #cross-entropy term
        if self.num_mixtures > 0:
            zsamples = torch.randn(nsamples,self.dim,device=self.device)
            samples = torch.sqrt(cov)*zsamples + mu
            cross_entropy_term= -self.current_logq(samples,
                                                   stabilizer=self.den_stabilizer).mean()
        else:
            cross_entropy_term = 0.0
        if verbose >= 3:
            print('loss terms ',logjoint_term.item(),lambd*self_entropy_term.item(),
                  cross_entropy_term.item())
        loss = -(logjoint_term + lambd*self_entropy_term + cross_entropy_term)
        return loss

    def _find_new_components(self,**kwargs):
        """
            Choose new boosting components. Arguments are explicited in update function
            The algorithm initial new component has mean sampled from current variational proposal,
            and variances sampled according to distrib_constant_nc. Then, the 
            RELBO is optimized, returning the component.
        """
        maxiter = kwargs.get("maxiter_nc",300)
        lr = kwargs.get("lr_nc",0.1)
        distrib_constant = kwargs.get("distrib_constant_nc",1.0)
        distrib_type = kwargs.get("distrib_type_nc","HN")
        verbose = kwargs.get("verbose",1)
        nsamples_nc = kwargs.get("n_samples_nc",100)
        if self.num_mixtures == 0:
            raise NotImplementedError
        else:
            mu0 = self._sample_from_current(1,flatten=True)
            if distrib_type == "HN":
                distrib = torch.distributions.HalfNormal
            elif distrib_type == "HC":
                distrib = torch.distributions.HalfCauchy
            cov0 = distrib(torch.tensor([distrib_constant],device=self.device)).\
                        sample((self.dim,)).flatten()
            raw_cov0 = utils.invsoftplus(cov0)
            theta0 = torch.cat([mu0,raw_cov0])
        theta0.requires_grad = True
        optimizer = torch.optim.Adam([theta0],lr=lr)
        for i in range(maxiter):
            optimizer.zero_grad()
            theta0.requires_grad = True
            loss = self._evaluate_loss(theta0,nsamples=nsamples_nc,
                                      verbose=verbose)
            if verbose >= 2:
                print(theta0)
                print(loss)
            loss.backward()
            optimizer.step()
        mu = theta0[:self.dim].detach()
        cov = (utils.softplus(theta0[self.dim:])).detach() #covariance
        if verbose >= 1:
            print(loss)
        return mu,cov
            
    def _find_alpha_new(self,mu_new,cov_new,**kwargs):
        if self.num_mixtures == 0:
            return torch.tensor(1.0)
        else:
            b = kwargs.get("b_sn",0.1)
            n_samples = kwargs.get("n_samples_sn",100)
            n_iter = kwargs.get("n_iter_sn",100)
            n_iter_min = kwargs.get("n_iter_min",100)
            alpha_tol = kwargs.get("alpha_tol",1e-4)
            max_alpha = kwargs.get("max_alpha",1.0)
            verbose = kwargs.get("verbose",1)
            lower_limit = 0.0+1e-12
            upper_limit = max_alpha-1e-12
            alpha_t = torch.tensor(lower_limit,device=self.device) #Minimum alpha
            k = 0
            mu_new = mu_new.detach()
            cov_new = cov_new.detach()
            #Use bmc to evaluate logjoint integrations
            normed_eval_logjoint_current = self.bmcmodel.evaluate_integral_mixture(self.mu_t,
                                                                            self.cov_t,
                                                                            self.weights,
                                                                            diag=True)
            normed_eval_logjoint_new = self.bmcmodel.evaluate_integral(mu_new,cov_new,diag=True)
            eval_logjoint_current = self.evals_std*normed_eval_logjoint_current + self.evals_mean
            eval_logjoint_new = self.evals_std*normed_eval_logjoint_new + self.evals_mean
            for i in range(n_iter):
                samples_current = self._sample_from_current(n_samples)
                samples_new = self._sample_from_new(mu_new,cov_new,n_samples)
                log_line_interp_current = self._line_interp_logq(samples_current,
                                                                 alpha_t,
                                                                 mu_new,
                                                                 cov_new).mean()
                log_line_interp_new = self._line_interp_logq(samples_new,
                                                                 alpha_t,
                                                                 mu_new,
                                                                 cov_new).mean()
                if verbose >= 3:
                    print(log_line_interp_current.mean())
                    print(log_line_interp_new.mean())
                    print(eval_logjoint_current.mean())
                    print(eval_logjoint_new.mean())
                    print("*")
                gamma_current = log_line_interp_current - eval_logjoint_current
                gamma_new = log_line_interp_new - eval_logjoint_new
                d1dkl = gamma_new - gamma_current
                k += 1
                delta_alpha = b*d1dkl/k
                alpha_t = alpha_t - delta_alpha
                if verbose >= 2:
                    print(alpha_t,delta_alpha,i)
                if verbose >= 3:
                    print((1-alpha_t)*(eval_logjoint_current - log_line_interp_current) + \
                          alpha_t*(eval_logjoint_new - log_line_interp_new),'ok')
                if alpha_t > upper_limit:
                    alpha_t = torch.tensor(upper_limit,device=self.device)
                    if i >= n_iter_min : break
                elif alpha_t < lower_limit:
                    alpha_t = torch.tensor(lower_limit,device=self.device)
                    if i >= n_iter_min : break
                if torch.abs(delta_alpha).item() < alpha_tol: #Convergence
                    if i >= n_iter_min : break
            alpha_t = alpha_t.detach()
            if verbose >= 1:
                print(alpha_t)
            return alpha_t

    def _initialize_vb(self,mu0,cov0,device,**kwargs):
        mu_t = mu0.reshape(1,-1)
        cov_t = cov0.reshape(1,-1)
        weights = torch.ones(1,device=device)
        self.distribution = utils.MixtureNormalDiagCov(mu_t,cov_t,weights)
        self.num_mixtures = 1

    def _initialize_bmc(self,samples,evals_norm,device,**kwargs):
        if self.bmc_type == "FM":
            constant0 = kwargs.get("constant0",None)
            print(constant0)
            if not constant0:
                if self.normalization_mode == "zeromax":
                    constant0 = self.numstab
                else:
                    constant0 = -math.inf
                kwargs["constant0"] = constant0
            self.bmcmodel = BMC_FM(self.dim,
                                   device=device,**kwargs)
        elif self.bmc_type == "QM":
            self.bmcmodel = BMC_QM(self.dim,device=device,**kwargs)
        elif self.bmc_type == "SFM":
            self.bmcmodel = BMC_SFM(self.dim,constant0=-math.inf,
                                    device=device,**kwargs)
        elif self.bmc_type == "SQM":
            self.bmcmodel = BMC_SQM(self.dim,device=device,**kwargs)
        else:
            raise NotImplementedError("No valid bmcmodel")
        self.bmcmodel.update_samples(samples,evals_norm,**kwargs)
        
    def _initialize_with_samples(self,samples,evaluations=None,**kwargs):
        """
            
        """
        if not utils.is_none(evaluations):
            evaluations = evaluations.clone()
        else:
            evaluations = self.evaluate_logjoint(samples,stabilize=False)
        if self.refstab == None or self.normalization_mode == "zeromax":
            self.refstab = torch.max(evaluations).item()
        if self.normalization_mode == "zeromax" and self.numstab==-math.inf:
            self.numstab = torch.min(evaluations).item() - self.refstab
        evaluations[evaluations < self.num_stabilizer] = self.num_stabilizer
        evals_norm,self.evals_mean,self.evals_std = \
            self._standardize(evaluations)
        return evals_norm

    def _line_interp_logq(self,theta,alpha,mu_new,cov_new):
        mu_expanded = torch.cat([self.mu_t,mu_new.reshape(1,-1)])
        cov_expanded = torch.cat([self.cov_t,cov_new.reshape(1,-1)])
        logqs_expanded = utils.logmvnbatch(theta,mu_expanded,cov_expanded)
        weights_expanded = torch.cat([(1-alpha)*self.weights,alpha.reshape(-1)])
        res = utils.logexpsum(logqs_expanded,weights_expanded)
        return res
    
    def _sample_from_current(self,nsamples=1,flatten=False):
        return self.distribution.sample(nsamples,flatten)
    
    def _sample_from_new(self,mu_new,cov_new,nsamples=1,flatten=False):
        samples = torch.sqrt(cov_new)*torch.randn(nsamples,self.dim,device=cov_new.device) + mu_new
        if flatten and nsamples == 1:
            samples = samples.flatten()
        return samples

    def _standardize(self,evaluations):
        if self.normalization_mode == "normalize":
            mean = torch.mean(evaluations,dim=0)
            std = torch.std(evaluations,dim=0)
        elif self.normalization_mode in [None,"none"]:
            mean = torch.zeros((1,),device=evaluations.device)
            std = torch.ones((1,),device=evaluations.device)
        elif self.normalization_mode == "zeromax":
            mean = torch.max(evaluations,dim=0)[0]
            std = torch.ones((1,),device=evaluations.device)
        else:
            raise NotImplementedError("Normalization mode not valid")
        normalized = (evaluations-mean)/std
        return normalized,mean,std

    def _update_dense_model(self,acq_fn,num_evals=100,
                            mode="optimizing",
                            lr=0.1,
                            verbose=1,acq_reg=1e-4):
        loss_fn = self._choose_acquisition_dense(acq_fn,acq_reg)
        #TODO : Throw these into functions
        if mode == "optimizing":
            Xnew = self.samples_proposal(1)
            Xnew.requires_grad_()
            optimizer = torch.optim.Adam([Xnew],lr=lr)
            for i in range(num_evals):
                optimizer.zero_grad()
                Xnew.requires_grad = True
                loss = loss_fn(Xnew)
                if verbose >= 2:
                    print(loss)
                loss.backward()
                optimizer.step()
            Xnew.requires_grad = False
        elif mode == "sampling":
            Xproposals = self.samples_proposal(100)
            loss_proposals = loss_fn(Xproposals)
            _,ind_new = torch.min(loss_proposals,dim=0)
            Xnew = Xproposals[ind_new,:]
        evals_new = self.evaluate_logjoint(Xnew)
        unscaled_prediction_new = self.bmcmodel.prediction(Xnew,cov="none")
        evals_new,evals_norm = self._adjust_new_evaluations(evals_new)
        self.bmcmodel.update_samples_iter(Xnew,evals_new)
        return evals_new,unscaled_prediction_new
    
    def _update_sparse_model(self,**kwargs):
        raise NotImplementedError

    def _adjust_new_evaluations(self,evals_new):
        assert torch.all(evals_new >= self.num_stabilizer)
        evals_new = (evals_new - self.evals_mean)/self.evals_std
        evals_norm = self.evals_norm.clone()
        evals_norm = torch.cat([evals_norm,evals_new],dim=0)
        return evals_new,evals_norm
    
    #Aquisition functions
    def _choose_acquisition_dense(self,acq_fn,acq_reg=1e-4):
        if acq_fn == "prospective":
            loss_fn = lambda Xnew : -self._prospective_prediction(Xnew,acq_reg)
        elif acq_fn == "uncertainty_sampling":
            loss_fn = lambda Xnew : -self._us_prediction(Xnew,acq_reg)
        elif acq_fn == "mmlt":
            loss_fn = lambda Xnew : -self._mmlt_prediction(Xnew,acq_reg)
        elif acq_fn == "mmlt_prospective":
            loss_fn = lambda Xnew : -self._mmlt_prospective_prediction(Xnew,acq_reg)
        else:
            raise ValueError
        return loss_fn

    def _prospective_prediction(self,theta,acq_reg=1e-4):
        pred_mean,pred_var = self.bmcmodel.prediction(theta,cov="diag")
        reg_factor = torch.exp(-(acq_reg/pred_var - 1)*(pred_var < acq_reg).float())
        res = self.current_q(theta)**2*pred_var*torch.exp(pred_mean)*reg_factor
        return res

    def _soft_prospective_prediction(self,theta,acq_reg=1e-4):
        #Not used
        pred_mean,pred_var = self.bmcmodel.prediction(theta,cov="diag")
        reg_factor = torch.exp(-(acq_reg/pred_var - 1)*(pred_var < acq_reg).float())
        res = self.current_q(theta)**2*pred_var*utils.softplus(pred_mean)*reg_factor
        return res

    def _us_prediction(self,theta,acq_reg=1e-4):
        _,pred_var = self.bmcmodel.prediction(theta,cov="diag")
        reg_factor = torch.exp(-(acq_reg/pred_var - 1)*(pred_var < acq_reg).float())
        res = self.current_q(theta)**2*pred_var*reg_factor
        return res
    
    def _mmlt_prediction(self,theta,acq_reg=1e-4):
        pred_mean,pred_var = self.bmcmodel.prediction(theta,cov="diag")
        reg_factor = torch.exp(-(acq_reg/pred_var - 1)*(pred_var < acq_reg).float())
        return (torch.exp(pred_var)-1)*torch.exp(pred_var+2*pred_mean)*reg_factor
    
    def _mmlt_prospective_prediction(self,theta,acq_reg=1e-4):
        pred_mean,pred_var = self.bmcmodel.prediction(theta,cov="diag")
        reg_factor = torch.exp(-(acq_reg/pred_var - 1)*(pred_var < acq_reg).float())
        res = self.current_q(theta)*\
              (torch.exp(pred_var)-1)*torch.exp(pred_var+2*pred_mean)*reg_factor
        return res
    
    #device change
    def change_device(self,device):
        raise NotImplementedError("Current implementation is faulty")
        