# -*- coding: utf-8 -*-
import math
import functools

import torch
import numpy as np

from . import utils


class _BMCBase(object):
    def __init__(self,dim,minvalue=1e-6,noise=1e-4,
                 fixed_noise=True,fixed_output=False,
                 proportional_jitter=False,kernel_function="RBF",
                 device="cpu",**kwargs):
        """
            Base class. Use either BMC_FM or BMC_QM
            Arguments:
                dim : dimension of the distribution
                minvalue : minimum value for l and theta. Default : 1e-6
                noise : value of the noise. Default: 1e-4
                fixed_noise : whether the noise is fixed in optimization. Default: True
                proportional_jitter : if noise is fixed, tells if jitter is to be 
                                      proportional (to mean(tr(K))) 
                                      or absolute. Default : False
                kernel_function : which kernel function to use. Default : "RBF".
                                  others are ("PMat",). All experimental
                device : which device. Default: "cpu"
        """
        self.dim = dim
        self._param_transform = functools.partial(utils.softplus,
                                                  minvalue=0.0)
        self._inv_param_transform = functools.partial(utils.invsoftplus,
                                                      minvalue=0.0)
        self.noise = torch.tensor(noise)
        self.fixed_noise = fixed_noise
        self.fixed_output = fixed_output
        self.proportional_jitter = proportional_jitter
        self.kernel_function = kernel_function
        if self.kernel_function == "PMat":
            self.matern_coef = kwargs.get("matern_coef",0.5)
            self.degree_hermite = kwargs.get("degree_hermite",25)
            x_herm,w_herm = np.polynomial.hermite.hermgauss(self.degree_hermite)
            self.x_herm = torch.tensor(x_herm).float()
            self.w_herm = torch.tensor(w_herm).float()
        self._intvectype = None
        self.device_type = device
        
    def evaluate_integral(self,mu=None,cov=None,retvar=False,diag=False):
        """
            Evaluate the BMC integral, with p(x) = N(x;mu,cov).
            Arguments:
                mu : If None, calculate with the last one used. Default: None
                cov : If None, calculate with the last one used. 
                      Shape depends on the truth value of diag. Default: None
                retvar : Whether to return the variance. Default: False
                diag : Whether cov is a diagonal matrix. Default: False
        """
        if utils.is_none(mu) or utils.is_none(cov):
            if not hasattr(self,"integral_vector") or \
                self._intvectype != "SINGLE":
                raise TypeError
        else:
            self.make_integral_vector(mu,cov,retvar,diag)
        if not hasattr(self,"weights_vector"):
            self.make_weights_vector()
        mean = torch.matmul(self.integral_vector,self.weights_vector) + \
                    self.mean_factor
        if not retvar:
            return mean
        else:
            varterm1 = self.vardetterm
            Lz = torch.triangular_solve(self.integral_vector.reshape(-1,1),
                                        self.kernel_chol_lower,
                                        upper=False)[0]
            var = varterm1 - torch.matmul(Lz.t(),Lz)
            return mean,var
            
    def evaluate_integral_mixture(self,mu,cov,weights,retvar=False,diag=False):
        """
            Evaluate the BMC integral, with 
                p(x) = sum(weights[i]*N(x;mu[i,:],cov_i[i,:(:)])).
            Arguments:
                mu : Component means
                cov : Component covariances
                weights : Component weights
                retvar : Whether to return the variance. Default: False
                diag : Whether cov is a diagonal matrix. Default: False
        """
        self.make_integral_vector_mixture(mu,cov,weights,retvar,diag) #self._intvectype will be "MIXTURE"
        if not hasattr(self,"weights_vector"):
            self.make_weights_vector()
        mean = torch.matmul(self.integral_vector,self.weights_vector) + self.mean_factor
        if not retvar:
            return mean
        else:
            varterm1 = self.vardetterm
            Lz = torch.triangular_solve(self.integral_vector.reshape(-1,1),
                                        self.kernel_chol_lower,
                                        upper=False)[0]
            var = varterm1 - torch.matmul(Lz.t(),Lz)
            return mean,var
            
    def evaluate_log_likelihood(self,from_zero=True,
                                training_space="gspace"):
        """
            from_zero : Whether there is need to recalculate the kernel_matrix
            training_space : Training space.
        """
        if training_space == "gspace": #g-space
            K = self.kfunc(self.samples,self.samples,symmetrize=True)
            m = self._mean(self.samples)
            evals = self.evaluations
        elif training_space == "fspace": #f-space
            m,K = self._fspace_m_and_k(self.samples)
            evals = torch.exp(self.evaluations)
        if from_zero:
            K = utils.jitterize(K,self.noise,self.proportional_jitter)
            L = torch.cholesky(K,upper=False)
        else:
            if training_space == "fspace":
                raise NotImplementedError("fspace training must be from zero")
            else:
                L = self.kernel_chol_lower
        y = evals - m
        Ly = torch.triangular_solve(y,L,upper=False)[0]
        term1 = -0.5*torch.sum(Ly**2)
        term2 = -torch.sum(torch.log(torch.diag(L)))
        term3 = -0.5*self.evaluations.shape[0]*math.log(2*math.pi)
        self._log_likelihood = term1 + term2 + term3
        return self._log_likelihood
        
    def make_kernel_and_cholesky(self):
        """
            Make kernel matrix and it's cholesky factor
        """
        K = self.kfunc(self.samples,self.samples,symmetrize=True)
        K = utils.jitterize(K,self.noise,self.proportional_jitter)
        self.kernel_matrix = K
        self.kernel_chol_lower = torch.cholesky(K,upper=False)
    
    def make_weights_vector(self,update=False):
        """
            Make the weight vector
        """
        if not hasattr(self,"kernel_matrix") or update:
            self.make_kernel_and_cholesky()
        y = self.evaluations - self._mean(self.samples)
        L = self.kernel_chol_lower
        weights_vector = utils.potrs(y,L,upper=False)
        self.weights_vector = weights_vector

    def make_integral_vector(self,mu,cov,calcvar=True,diag=False):
        """
            Makes the integral vector
            X : (n,d) tensor
            theta : 0d tensor or float
            l : (d,) tensor
            mu : (d,) tensor
            cov : (d,d) tensor or (d,) tensor (depending on diag)
            outputs (n,) tensor and 0d tensor
        """
        
        #Covariance term
        l = self.lengthscale
        theta = self.outputscale
        X = self.samples
        if not diag:
            integral_vector = self._zvec(mu,cov,X,theta,l)
        else:
            integral_vector = self._zvecdiag(mu,cov,X,theta,l)
        self.integral_vector = integral_vector
        self.mean_factor = self._get_mean_factor(mu,cov,diag)
        if calcvar:
            if not diag:
                self.vardetterm = self._vardetterm(cov,theta,l)
            else:
                self.vardetterm = self._vardettermdiag(cov,theta,l)
        #Covariance constant
        self._intvectype = "SINGLE"
        return self.integral_vector,self.mean_factor
    
    def make_integral_vector_mixture(self,mu,cov,weights,calcvar=False,diag=False):
        """
            integral vector and mean for a mixture of t gaussians
            mu : (t,d) tensor
            cov : (t,d,d) tensor or (t,d) tensor (depending on diag)
            weights : (t,) tensor
            outputs (n,) tensor and 0d tensor
        """
        l = self.lengthscale #(d,)
        theta = self.outputscale #(,)
        X = self.samples #(n,d)
        if not diag:
            zvec = self._zvecmvn(mu,cov,X,theta,l,weights)
        else:
            zvec = self._zvecmvndiag(mu,cov,X,theta,l,weights)
        #Mean term
        self.integral_vector = zvec
        self.mean_factor = self._get_mean_factor_mixture(mu,cov,weights,diag)
        if calcvar:
            if not diag:
                self.vardetterm = self._vardettermmvn(mu,cov,theta,l,weights)
            else:
                self.vardetterm = self._vardettermmvndiag(mu,cov,theta,l,weights)
        self._intvectype = "MIXTURE"
        return self.integral_vector,self.mean_factor
    
    def prediction(self,X,cov="diag"):
        """
            X : (n_eval,ndim) tensor
            returns:
                if cov="none" : mean
                if cov="diag" : mean,var
                if cov="full" : mean,cov
        """
#        raise NotImplementedError
        m = self._mean(X)
        kx = self.kfunc(X,self.samples)
        if cov == "none" and hasattr(self,"weights_vector"):
            return m + torch.matmul(kx,self.weights_vector)
        else: #Dumb innefective way for now
            L = self.kernel_chol_lower
            y = self.evaluations - self._mean(self.samples) #(nsamples,1)
            pred_mean = m + utils.inv_quad_chol_lower(kx,L,y) #(eval,1)
            if cov == "none":
                return pred_mean
            else:
                kxx = self.kfunc(X,X,symmetrize=True)
                if cov == "diag":
                    pred_var = torch.diag(kxx).reshape(-1,1) - \
                               utils.inv_quad_chol_lower_3(L,kx.transpose(1,0)) #(n_eval,1)
                else:
                    pred_var = kxx - utils.inv_quad_chol_lower_2(L,kx.transpose(1,0)) #(n_eval,n_eval)
                return pred_mean,pred_var
        
    def optimize_model(self,lr=0.5,training_iter=50,**kwargs):
        """
            Optimize the log likelihood of the data
        """
        verbose = kwargs.get("verbose",1)
        training_space = kwargs.get("training_space","gspace")
        output_reg = kwargs.get("output_reg",None)
        for p in self._params():
            p.requires_grad_()
        params = self._params()
        if self.fixed_output:
            params = params[1:]
        if not self.fixed_noise:
            self._raw_noise.requires_grad_()
            params.append(self._raw_noise)
        optimizer = torch.optim.Adam(params,lr=lr)
        for i in range(training_iter):
            optimizer.zero_grad()
            loss = -self.evaluate_log_likelihood(from_zero=True,
                                                 training_space=training_space)
            if output_reg and not self.fixed_output:
                loss += output_reg*self.outputscale**2
            loss.backward()
            optimizer.step()
            if (i+1)%1 == 0 and verbose >= 1:
                print("Step %i, %f"%(i+1,loss.item()))
                if verbose >= 2:
                    print(params)
#                if verbose >= 3:
#                    for p in self._params(): print(p.grad)
        for p in self._params():
            if verbose >= 3:
                print(p.grad)
            p.requires_grad = False
        if not self.fixed_noise:
            self._raw_noise.requires_grad = False

    def update_samples(self,samples,evaluations,**kwargs):
        """
            Update the samples to samples,evaluations
        """
        assert samples.shape[1] == self.dim
        assert samples.device.type == self.device_type
        self.samples = samples
        self.evaluations = evaluations
        if kwargs.get("empirical_init",False):
            raise NotImplementedError
            self._set_init_params_empirically()
    
    def update_samples_iter(self,new_samples,new_evaluations,
                            update_all_evaluations=False):
        """
            Update the kernel matrix iteratively
            new_samples : (m,d)
            new_evaluations : (m,1)
        """
        Knew,Lnew = self._updated_kernel_and_chol(new_samples)
        self.kernel_matrix = Knew
        self.kernel_chol_lower = Lnew
        self.samples = torch.cat([self.samples,new_samples],dim=0)
        if not update_all_evaluations:
            self.evaluations = torch.cat([self.evaluations,new_evaluations],dim=0)
        else:
            self.evaluations = new_evaluations
        self.make_weights_vector()
        
    @property
    def lengthscale(self):
        return self._param_transform(self._raw_lengthscale)
    
    @property
    def outputscale(self):
        return self._param_transform(self._raw_outputscale)
    
    @property
    def noise(self):
        return self._param_transform(self._raw_noise)
        
    @noise.setter
    def noise(self,value):
        self._raw_noise = self._inv_param_transform(value)

    def _set_init_params(self,theta0,l0,center0,lmeans0,constant0):
        raise NotImplementedError
    
    def _set_init_params_empirically(self):
        raise NotImplementedError

    def _params(self):
        raise NotImplementedError
    
    def _mean(self,x):
        raise NotImplementedError
    
    def _get_mean_factor(self,mu,cov):
        raise NotImplementedError
        
    def _get_mean_factor_mixture(self,mu,cov,weights):
        raise NotImplementedError
    
    def _zvec(self,mu,cov,X,theta,l):
        if self.kernel_function == "RBF":
            C = cov + torch.diag(l**2)
            L = torch.cholesky(C,upper=False)
            Xm = X - mu #nxd#
            LX = torch.triangular_solve(Xm.transpose(1,0),L,upper=False)[0] #d x n
            expoent = -0.5*torch.sum(LX**2,dim=0) #(n,)
            det = torch.prod(1/l**2)*torch.prod(torch.diag(L))**2 #|I + A^-1B|
            zvec = theta/torch.sqrt(det)*torch.exp(expoent) #(n,)
        else:
            raise NotImplementedError
        return zvec
    
    def _zvecdiag(self,mu,cov,X,theta,l):
        if self.kernel_function == "RBF":
            C = cov + l**2 #(d,)
            Xm = X - mu #(nxd)
            expoent = -0.5*torch.sum(Xm**2/C,dim=1)
            det = torch.prod(1/l**2*C)
            zvec = theta/torch.sqrt(det)*torch.exp(expoent) #(n,)
        elif self.kernel_function == "PMat":
            mu_ = mu.unsqueeze(-1) #(d,1)
            cov_ = cov.unsqueeze(-1) #(d,1)
            X_ = X.unsqueeze(-1) #(n,d,1)
            l_ = l.unsqueeze(-1) #(d,1)
            X2_ = torch.sqrt(2*cov_)*self.x_herm + mu_ #(d,h)
            norm_ = torch.abs(X_ - X2_) #(n,d,h)
            R = norm_/l_ #(n,d,h)
            if self.matern_coef == 0.5:
                K1 = torch.exp(-R) #(t,n,d,h)
            elif self.matern_coef == 1.5:
                K1 = (1 + math.sqrt(3)*R)*torch.exp(-math.sqrt(3)*R)
            elif self.matern_coef == 2.5:
                K1 = (1 + math.sqrt(5)*R + 5.0/3.0*R**2)*torch.exp(-math.sqrt(5)*R)
            else:
                raise NotImplementedError
            K2 = 1/math.sqrt(math.pi)*torch.sum(self.w_herm*K1,dim=-1) #(t,n,d)
            zvec = theta*torch.prod(K2,dim=-1) #(t,n)
        else:
            raise NotImplementedError
        return zvec
        
    def _zvecmvn(self,mu,cov,X,theta,l,weights):
        if self.kernel_function == "RBF":
            d = l.numel()
            t = weights.numel()
            C = cov + torch.diag(l**2).repeat([t,1,1]) #(t,d,d)
            L = torch.cholesky(C,upper=False) #(t,d,d)
            Xm = X.repeat([t,1,1]) - mu.reshape([t,1,d]) #(t,n,d)
            LX = utils.batch_trtrs(Xm.transpose(-2,-1),L,upper=False) #(t,d,n)
            expoent = -0.5*torch.sum(LX**2,dim=1) #(t,n)
            det = torch.prod(1/l**2)*\
                  torch.prod(utils.batch_diag1(L),dim=1,keepdim=True)**2 #|I + A^-1B| (t,1)
            vec_ = theta/torch.sqrt(det)*torch.exp(expoent) #(t,n)
            zvec = (weights.reshape(-1,1)*vec_).sum(dim=0) #(n,)
        else:
            raise NotImplementedError
        return zvec

    def _zvecmvndiag(self,mu,cov,X,theta,l,weights):
        if self.kernel_function == "RBF":
            # mu : (t,d)
            # cov : (t,d)
            # X : (n,d)
            # theta: scalar
            # l : (d,)
            C = (cov + l**2).unsqueeze(1) #(t,1,d)
            Xm = X.unsqueeze(0) - mu.unsqueeze(1) #(t,n,d)
            expoent = -0.5*torch.sum(Xm**2/C,dim=-1) #(t,n)
            det = torch.prod(1/l**2*C,dim=-1) #(t,1)
            vec_ = theta/torch.sqrt(det)*torch.exp(expoent) #(t,n)
            zvec = (weights.reshape(-1,1)*vec_).sum(dim=0) #(n,)
        elif self.kernel_function == "PMat":
            mu_ = mu.unsqueeze(-1).unsqueeze(1) #(t,1,d,1)
            cov_ = cov.unsqueeze(-1).unsqueeze(1) #(t,1,d,1)
            X_ = X.unsqueeze(-1) #(n,d,1)
            l_ = l.unsqueeze(-1) #(d,1)
            X2_ = torch.sqrt(2*cov_)*self.x_herm + mu_ #(t,1,d,h)
            norm_ = torch.abs(X_ - X2_) #(t,n,d,h)
            R = norm_/l_ #(n,d,h)
            if self.matern_coef == 0.5:
                K1 = torch.exp(-R) #(t,n,d,h)
            elif self.matern_coef == 1.5:
                K1 = (1 + math.sqrt(3)*R)*torch.exp(-math.sqrt(3)*R)
            elif self.matern_coef == 2.5:
                K1 = (1 + math.sqrt(5)*R + 5.0/3.0*R**2)*torch.exp(-math.sqrt(5)*R)
            else:
                raise NotImplementedError
            K2 = 1/math.sqrt(math.pi)*torch.sum(self.w_herm*K1,dim=-1) #(t,n,d)
            vec_ = theta*torch.prod(K2,dim=-1) #(t,n)
            zvec = (weights.reshape(-1,1)*vec_).sum(dim=0) #(n,)
        else:
            raise NotImplementedError
        return zvec
    
    def _vardetterm(self,cov,theta,l):
        if self.kernel_function == "RBF":
            Ccov = cov + 0.5*torch.diag(l**2)
            Lcov = torch.cholesky(Ccov,upper=False)
            detcov = torch.prod(2/(l**2))*torch.prod(torch.diag(Lcov))**2
            term = theta/torch.sqrt(detcov)
        else:
            raise NotImplementedError
        return term

    def _vardettermdiag(self,cov,theta,l):
        if self.kernel_function == "RBF":
            Ccov = cov + 0.5*(l**2)
            detcov = torch.prod(2/(l**2)*Ccov)
            term = theta/torch.sqrt(detcov)
        else:
            raise NotImplementedError
        return term

    def _vardettermmvn(self,mu,cov,theta,l,weights):
        raise NotImplementedError

    def _vardettermmvndiag(self,mu,cov,theta,l,weights):
        if self.kernel_function == "RBF":
            # mu : (t,d)
            # cov : (t,d)
            # l : (d,)
            # theta : (,)
            # weights : (t,)
            mu1_ = mu.unsqueeze(0) #(1,t,d)
            mu2_ = mu.unsqueeze(1) #(t,1,d)
            cov1_ = cov.unsqueeze(0) #(1,t,d)
            cov2_ = cov.unsqueeze(1) #(t,1,d)
            covsum = cov1_ + cov2_ #(t,t,d)
            meandiff = mu1_ - mu2_ #(t,t,d)
            expterm = (meandiff**2)/(covsum+l) #(t,t,d)
            term1 = torch.exp(-0.5*torch.sum(expterm,dim=-1)) #(t,t)
            detterm = 1.0 + covsum/l #(t,t,d)
            term2 = theta/torch.sqrt(torch.prod(detterm,dim=-1)) #(t,t)
            term3 = torch.ger(weights.flatten(),weights.flatten()) #(t,t)
            result = torch.sum(term1*term2*term3)
        else:
            raise NotImplementedError
        return result

    def _updated_kernel_and_chol(self,Xnew):
        kxs = self.kfunc(Xnew,self.samples)
        kxx = self.kfunc(Xnew,Xnew,symmetrize=True)
        kxx = utils.jitterize(kxx,self.noise,self.proportional_jitter)
        Knew = utils.expand(self.kernel_matrix,kxs.transpose(1,0),kxx)
        Lnew = utils.expand_cholesky_lower(self.kernel_chol_lower,kxs,kxx)
        return Knew,Lnew
            
    def _new_variance(self,Xnew,mu=None,cov=None):
        #Uses current mu as reference
        if utils.is_none(mu) or utils.is_none(cov):
            if self._intvectype != "SINGLE":
                raise TypeError
            else:
                from_current_mucov = True
        else:
            from_current_mucov = False
        #Xnew : (n+k,d)
        #1 : expand zvec
        l = self.lengthscale #(d,)
        theta = self.outputscale #(n,)
        _,Lnew = self._updated_kernel_and_chol(Xnew)
        if from_current_mucov:
            zvecnew = self._zvec(mu,cov,Xnew,theta,l)
            zvecstar = torch.cat([self.integralvector,zvecnew],dim=0) #(n+k,)
            varterm1 = self.vardetterm
        else:
            Xstar = torch.cat([self.samples,Xnew],dim=0)
            zvecstar = self._zvec(mu,cov,Xstar,theta,l)
            varterm1 = self._vardetterm(cov,theta,l)
        Lnewzvecstar = torch.triangular_solve(zvecstar.reshape(-1,1),
                                              Lnew,upper=False)[0] #(n+k,1)
        varterm2 = torch.matmul(Lnewzvecstar.t(),Lnewzvecstar)
        return varterm1 - varterm2

    def _persistent_names(self):
        names = ["samples","evaluations","inducing",
                 "jitter","Luu","Ddiag","Kuf","LBWdt",
                 "weights_vector","integral_vector",
                 "vardetterm","_raw_noise","kernel_matrix","kernel_chol_lower",
                 "_raw_outputscale","_raw_lengthscale",
                 "_raw_center","_raw_mean_lengthscale","_raw_mean_constant"]
        return names
    
    def _fspace_m_and_k(self,X):
        m_ = self._mean(X)
        K_ = self.kfunc(X,X,symmetrize=True)
        m = torch.exp(m_ + 0.5*torch.diag(K_).reshape(-1,1))
        K = (torch.exp(K_)-1)*(m@m.t())
        return m,K
        
        
    def kfunc(self,X,Y,symmetrize=False,diag=False):
        if self.kernel_function == "RBF":
            return utils.rbf_kernel(X,Y,self.outputscale,
                                    self.lengthscale,symmetrize=symmetrize,
                                    diag=diag)
        elif self.kernel_function == "PMat":
            return utils.prod_matern_kernel(X,Y,self.outputscale,
                                    self.lengthscale,self.matern_coef,
                                    symmetrize=symmetrize,
                                    diag=diag)
        else:
            raise NotImplementedError
        

    #change device
    def change_device(self,device):
        for name in self._persistent_names():
            if hasattr(self,name) and torch.is_tensor(getattr(self,name)):
                setattr(self,name,getattr(self,name).to(device))
        self.device_type = "device"


class BMC_QM(_BMCBase):
    def __init__(self,dim,minvalue=1e-6,
                 theta0=1.0,l0=1.0,
                 center0=0.0,lmeans0=10.0,constant0=0.0,
                 noise=1e-4,device="cpu",**kwargs):
        """
            dim : dimension of the distribution
        """
        super().__init__(dim,minvalue,noise,device=device,**kwargs)
        self._set_init_params(theta0,l0,center0,lmeans0,constant0,device)        
    
    @property
    def mean_center(self):
        return self._raw_center
    
    @property
    def mean_lengthscale(self):
        return self._param_transform(self._raw_mean_lengthscale)
    
    @property
    def mean_constant(self):
        return self._raw_mean_constant
    
    def _set_init_params(self,theta0,l0,center0,lmeans0,constant0,
                         device="cpu"):
        if theta0 == "default":
            theta0 = 1.0
        if l0 == "default":
            l0 = 1.0
        def tensorize(p):
            p_ = torch.tensor(p,device=device)
            if p_.numel() == 1:
                p_ = p_*torch.ones(self.dim,device=device)
            else:
                assert p_.numel() == self.dim
                p_ = p_.flatten()
            return p_
        self._raw_outputscale = self._inv_param_transform(torch.tensor(theta0,device=device))
        self._raw_lengthscale = self._inv_param_transform(tensorize(l0))
        self._raw_center = tensorize(center0)
        self._raw_mean_lengthscale = self._inv_param_transform(tensorize(lmeans0))
        self._raw_mean_constant = torch.tensor(constant0,device=device)
    
    def _set_init_params_empirically(self,device="cpu"):
        if device != "cpu":
            raise NotImplementedError
        assert hasattr(self,"samples") and hasattr(self,"evaluations")
        #Set mean center
        mean_lengthscale,mean_center,mean_constant = \
            utils.quadratic_mean_lsq(self.samples,self.evaluations)
        self._raw_center = mean_center
        self._raw_mean_lengthscale = self._inv_param_transform(mean_lengthscale)
        self._raw_mean_constant = mean_constant
        raw_mean_constant,raw_center_index = torch.max(self.evaluations,dim=0)
        outputscale = torch.std(self.evaluations - self._mean(self.samples))**2
        self._raw_outputscale = self._inv_param_transform(outputscale)
        lengthscale = torch.std(self.samples,dim=0)
        self._raw_lengthscale = self._inv_param_transform(lengthscale)

    def _params(self):
        return [self._raw_outputscale,self._raw_lengthscale,
                self._raw_center,self._raw_mean_lengthscale,self._raw_mean_constant]
    
    def _mean(self,x):
        x_ = (x-self.mean_center)/self.mean_lengthscale
        return -0.5*torch.sum(x_**2,dim=1,keepdim=True) + self.mean_constant

    def _get_mean_factor(self,mu,cov,diag=False): #Only right for diagonal cov
        if not diag:
            cov_ = torch.diag(cov)
        else:
            cov_ = cov
        lmean = self.mean_lengthscale
        center = self.mean_center
        mean_factor = -0.5*(torch.sum(cov_/(lmean**2)) + \
                            torch.sum((mu-center)**2/(lmean**2))) + self.mean_constant
        return mean_factor
    
    def _get_mean_factor_mixture(self,mu,cov,weights,diag=False): #Only right for diagonal cov
        if not diag:
            cov_ = utils.batch_diag1(cov)
        else:
            cov_ = cov
        lmean = self.mean_lengthscale
        center = self.mean_center
        mean_term_1 = -0.5*torch.sum(cov_/(lmean**2),dim=1) #(t,)
        mean_term_2 = -0.5*torch.sum((mu-center)**2/(lmean**2),dim=1) #(t,)
        mean_factor = torch.sum(weights*(mean_term_1+mean_term_2)) + self.mean_constant        
        return mean_factor
    
    def _persistent_names(self):
        names = ["samples","evaluations","inducing",
                 "jitter","kernel_matrix","kernel_chol_lower",
                 "weights_vector","integral_vector",
                 "vardetterm","_raw_noise",
                 "_raw_outputscale","_raw_lengthscale",
                 "_raw_mean_center","_raw_mean_lengthscale","_raw_mean_constant"]
        return names
    

class BMC_FM(_BMCBase):
    def __init__(self,dim,minvalue=1e-6,
                 theta0="default",l0="default",
                 constant0=-math.inf,scale_min=1.0,
                 noise=1e-4,device="cpu",**kwargs):
        """
            dim : dimension of the distribution
            constant0 : The mean constant, if not -math.inf. 
                        If -math.inf, schema here (experimental) 
                        is that if constant is -math.inf,
                        we get a very low constant, 
                        min_eval - scale_min*(max_val - min_eval)
        """
        super().__init__(dim,minvalue,noise,device=device,**kwargs)
        self.scale_min = scale_min
        self._set_init_params(theta0,l0,constant0,device)        
        
    @property
    def mean_constant(self):
        if self._raw_mean_constant != -math.inf:
            return self._raw_mean_constant
        else:
            eval_max = max(self.evaluations)
            eval_min = min(self.evaluations)
            D = self.scale_min
            return eval_min - D*(eval_max - eval_min)
    
    @mean_constant.setter
    def mean_constant(self,value):
        if value != -math.inf:
            self._raw_mean_constant = torch.tensor(value)
        else:
            self._raw_mean_constant = -math.inf
        
    def _set_init_params(self,theta0,l0,constant0,device="cpu"):
        if theta0 == "default":
            if constant0 != -math.inf and constant0 < 0:
                theta0 = constant0/-2.0
            else:
                theta0 = 1.0
        if l0 == "default":
            l0 = 1.0
        def tensorize(p):
            p_ = torch.tensor(p,device=device)
            if p_.numel() == 1:
                p_ = p_*torch.ones(self.dim,device=device)
            else:
                assert p_.numel() == self.dim
                p_ = p_.flatten()
            return p_
        self._raw_outputscale = self._inv_param_transform(torch.tensor(theta0,device=device))
        self._raw_lengthscale = self._inv_param_transform(tensorize(l0))
        if constant0 != -math.inf:
            self._raw_mean_constant = torch.tensor(constant0)
        else:
            self._raw_mean_constant = -math.inf
        
    def _params(self):
        return [self._raw_outputscale,self._raw_lengthscale]
    
    def _mean(self,x):
        return self.mean_constant*torch.ones((x.shape[0],1),
                                             device=x.device)

    def _get_mean_factor(self,mu,cov,diag):
        mean_factor = self.mean_constant
        return mean_factor
    
    def _get_mean_factor_mixture(self,mu,cov,weights,diag):
        mean_factor = self.mean_constant      
        return mean_factor