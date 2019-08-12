# -*- coding: utf-8 -*-
import math
import functools

import torch
import numpy as np

from . import utils


class _BMCSparseBase(object):
    def __init__(self,dim,approx_type="VFE",
                 minvalue=1e-6,jitter=1e-4,noise=1e-2,
                 fixed_noise = True,device="cpu"):
        """
            dim : dimension of the distribution
            Uses VFE for approximation
        """
        self.dim = dim
        self.approx_type = approx_type
        self._param_transform = functools.partial(utils.softplus,minvalue=0.0)
        self._inv_param_transform = functools.partial(utils.invsoftplus,minvalue=0.0)
        self.jitter = torch.tensor(jitter)
        self.noise = torch.tensor(noise)
        self.fixed_noise = fixed_noise
        self.device_type = device
        
    def evaluate_integral(self,mu=None,cov=None,retvar=False,diag=False):
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
            LuuKuf = torch.triangular_solve(self.Kuf,self.Luu,upper=False)[0] #(u,f)
            LuuKuz = torch.triangular_solve(self.integral_vector.reshape(-1,1),
                                            self.Luu,upper=False)[0] #(u,x)
            Qzf = torch.matmul(LuuKuz.t(),LuuKuf)
            varterm2 = torch.matmul(Qzf,Qzf.t())/self.noise #Substitute when FITC
            varterm3 = torch.matmul(torch.matmul(Qzf,self.LBWdt.t()),
                                       torch.matmul(self.LBWdt,Qzf.t()))
            var = varterm1 - (varterm2 - varterm3)
            return mean,var
            
    def evaluate_integral_mixture(self,mu,cov,weights,retvar=False,diag=False):
        self.make_integral_vector_mixture(mu,cov,weights,diag) #self._intvectype will be "MIXTURE"
        if not hasattr(self,"weights_vector"):
            self.make_weights_vector()
        mean = torch.matmul(self.integral_vector,self.weights_vector) + self.mean_factor
        if not retvar:
            return mean
        else:
            raise NotImplementedError
    
    def evaluate_log_likelihood(self,from_zero=True,warped=True):
        device = self.samples.device
        if warped:
            y = self.evaluations - self._mean(self.samples) #(N,1)
            Kuu = self.kfunc(self.inducing,self.inducing)
            Kuf = self.kfunc(self.inducing,self.samples)
        else:
            meany,_ = self._unwarped_k_and_m(self.samples)
            y = torch.exp(self.evaluations) - self._mean(self.samples) #(N,1)
            _,Kuu = self._unwarped_k_and_m(self.inducing)
            Kuf = self._unwarped_k_2(self.inducing,self.samples)
        if from_zero:
            N = self.samples.shape[0]
            Kuu = utils.jitterize(Kuu,self.jitter)
            Luu = torch.cholesky(Kuu,upper=False) #(M,M)
            W = torch.triangular_solve(Kuf,Luu,upper=False)[0].t() #(N,M) #W W^T = Qff
            Ddiag = self.noise*torch.ones(N,device=device) #(N,)
            if self.approx_type == "FITC":
                diagQff = (W*W.t()).sum(dim=1,keepdim=True)
                diagKff = utils.rbf_kernel(self.samples,self.samples,
                                           theta=self.outputscale,
                                           l=self.lengthscale,
                                           diag=True)
                Ddiag += diagKff - diagQff
            if self.approx_type == "VFE":#trace_term
                Kffdiag = self.outputscale*torch.ones(N,device=device) #(N,)
                Qffdiag = W.pow(2).sum(dim=-1) #(N,)
                trace_term = (Kffdiag - Qffdiag).sum()/self.noise
        else:
            raise NotImplementedError
        Wd = W/Ddiag.reshape(-1,1) #(N,M) D^-1 W
        B = utils.jitterize(torch.matmul(W.t(),Wd),1.0) #(M,M)
        LB = torch.cholesky(B,upper=False) #(M,M)
        # D^-1 - Wd Lb^-T Lb^-1 Wd^T = (Qff + D)^{-1}
        WDy = torch.matmul(Wd.t(),y) #(M,1)
        LBDy = torch.triangular_solve(WDy,LB,upper=False)[0] #(M,1)
        term1a = torch.sum(y**2/(Ddiag.reshape(-1,1))) #(,)
        term1b = torch.sum(LBDy**2) #(,)
        term1 = -0.5*(term1a - term1b)
        term2a = 2*torch.sum(torch.log(torch.diag(LB)))
        term2b = torch.sum(torch.log(Ddiag))
        term2 = -0.5*(term2a + term2b)
        term3 = -0.5*self.evaluations.shape[0]*math.log(2*math.pi)
        if self.approx_type == "VFE":
            term4 = -0.5*trace_term
            result = term1 + term2 + term3 + term4
        else:
            result = term1 + term2 + term3
        return result
                
    def make_kernel_and_cholesky(self):
        """
            Make Kmm and Omega = (Epsilon^-1) (according to Titsias)
        """
        device = self.samples.device
        N = self.samples.shape[0]
        Kuu = utils.rbf_kernel(self.inducing,self.inducing,
                                        self.outputscale,
                                        self.lengthscale)
        Kuf = utils.rbf_kernel(self.inducing,self.samples,
                                        self.outputscale,
                                        self.lengthscale)
        Kuu = utils.jitterize(Kuu,self.jitter)
        Luu = torch.cholesky(Kuu,upper=False) #(M,M)
        W = torch.triangular_solve(Kuf,Luu,upper=False)[0].t() #(N,M) #W W^T = Qff
        Ddiag = self.noise*torch.ones(N,device=device) #(N,)
        if self.approx_type == "FITC":
            diagQff = (W*W.t()).sum(dim=1,keepdim=True)
            diagKff = utils.rbf_kernel(self.samples,self.samples,
                                       theta=self.outputscale,
                                       l=self.lengthscale,
                                       diag=True)
            Ddiag += diagKff - diagQff
        Wd = (W/Ddiag.reshape(-1,1)) #(N,M)
        B = utils.jitterize(torch.matmul(W.t(),Wd),1.0) #(M,M) #(I + W^T W)
        LB = torch.cholesky(B,upper=False) #(M,M)
        # D^-1 - Wd Lb^-T Lb^-1 Wd^T = (Qff + D)^{-1}
        LBWdt = torch.triangular_solve(Wd.t(),LB,upper=False)[0] #(M,N)
        self.Ddiag = Ddiag
        self.LBWdt = LBWdt #D^{-1} - (LBWdt)^T(LBWdt) = (Qff + D)^{-1}
        self.Luu = Luu
        self.Kuf = Kuf
        
    def make_weights_vector(self,update=False):
        if not hasattr(self,"Luu") or not hasattr(self,"LBWdt") or update:
            self.make_kernel_and_cholesky()
        y = self.evaluations - self._mean(self.samples)
        LBWdty = torch.matmul(self.LBWdt,y) #(M,1)
        Qffy1 = y/self.Ddiag.reshape(-1,1) #(N,1)
        Qffy2 = torch.matmul(self.LBWdt.t(),LBWdty)
        Qffy = Qffy1 - Qffy2 #Qff^{-1}y
        KufQffy = torch.matmul(self.Kuf,Qffy)
        weights_vector = utils.potrs(KufQffy,self.Luu,upper=False)
        self.weights_vector = weights_vector
        
    def make_integral_vector(self,mu,cov,calcvar=True,diag=False):
        """
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
        Xu = self.inducing
        if not diag:
            integral_vector = self._zvec(mu,cov,Xu,theta,l)
        else:
            integral_vector = self._zvecdiag(mu,cov,Xu,theta,l)
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
    
    def make_integral_vector_mixture(self,mu,cov,weights,diag=False):
        """
            integral vector and mean for a mixture of t gaussians
            mu : (t,d) tensor
            cov : (t,d,d) tensor or (t,d) tensor (depending on diag)
            weights : (t,) tensor
            outputs (n,) tensor and 0d tensor
        """
        l = self.lengthscale #(d,)
        theta = self.outputscale #(,)
        Xu = self.inducing #(n,d)
        if not diag:
            zvec = self._zvecmvn(mu,cov,Xu,theta,l,weights)
        else:
            zvec = self._zvecmvndiag(mu,cov,Xu,theta,l,weights)
        #Mean term
        self.integral_vector = zvec
        self.mean_factor = self._get_mean_factor_mixture(mu,cov,weights,diag)
        self.vardetterm = None
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
        m = self._mean(X)
        kxu = utils.rbf_kernel(X,self.inducing,
                                    self.outputscale,
                                    self.lengthscale,
                                    symmetrize=False) #(n_eval,nsamples)
        if cov == "none" and hasattr(self,"weights_vector"):
            return m + torch.matmul(kxu,self.weights_vector)
        else: #Dumb innefective way for now
            y = self.evaluations - self._mean(self.samples) #(nsamples,1)
            #Make Qxf
            LuuKuf = torch.triangular_solve(self.Kuf,self.Luu,upper=False)[0] #(u,f)
            LuuKux = torch.triangular_solve(kxu.t(),self.Luu,upper=False)[0] #(u,x)
            Qxf = torch.matmul(LuuKux.t(),LuuKuf)
            pred_mean_1 = torch.matmul(Qxf,y)/self.noise #Substitute when FITC
            pred_mean_2 = torch.matmul(torch.matmul(Qxf,self.LBWdt.t()),
                                       torch.matmul(self.LBWdt,y))
            pred_mean = m + pred_mean_1 - pred_mean_2 #(eval,1)
            if cov == "none":
                return pred_mean
            else:
                kxx = utils.rbf_kernel(X,X,
                                              self.outputscale,
                                              self.lengthscale,
                                              symmetrize=False) #(n_eval,n_eval)
                pred_var_2 = torch.matmul(Qxf,Qxf.t())/self.noise #Substitute when FITC
                pred_var_3 = torch.matmul(torch.matmul(Qxf,self.LBWdt.t()),
                                           torch.matmul(self.LBWdt,Qxf.t()))
                pred_var = kxx - (pred_var_2 - pred_var_3)
                if cov == "diag":
                    pred_var = torch.diag(pred_var).reshape(-1,1)
                return pred_mean,pred_var
    
    def prospective_prediction(self,X):
        pred_mean,pred_var = self.prediction(X,cov="diag")
        res = pred_var*torch.exp(pred_mean)
        return res
    
    def optimize_model(self,lr=0.5,training_iter=50,**kwargs):
        verbose = kwargs.get("verbose",True)
        update_inducing = kwargs.get("update_inducing",False)
        warped = kwargs.get("warped",True)
        for p in self._params():
            p.requires_grad_()
        params = self._params()
        if update_inducing:
            self.inducing.requires_grad_()
            params.append(self.inducing)
        if not self.fixed_noise:
            self._raw_noise.requires_grad_()
            params.append(self._raw_noise)
        optimizer = torch.optim.Adam(params,lr=lr)
        for i in range(training_iter):
            optimizer.zero_grad()
            loss = -self.evaluate_log_likelihood(from_zero=True,
                                                 warped=warped)
            loss.backward()
            optimizer.step()
            if (i+1)%1 == 0 and verbose:
                print("Step %i, %f"%(i+1,loss.item()))
        for p in self._params():
            p.requires_grad = False
        if update_inducing:
            self.inducing.requires_grad = False
        if not self.fixed_noise:
            self._raw_noise.requires_grad = False

    def update_samples(self,samples,evaluations,ninducing,**kwargs):
        assert samples.shape[1] == self.dim
        assert samples.device.type == self.device_type
        self.samples = samples
        self.evaluations = evaluations
        #Inducing points
        if kwargs.get("choose_inducing_points",True):
            self._choose_inducing_points(self.samples,ninducing)
            if torch.any(torch.isnan(self.inducing)):
                raise ValueError("K-means returned none samples")
        else:
            assert hasattr(self,"inducing")
        if kwargs.get("empirical_init",False):
            self._set_init_params_empirically()
    
    def update_samples_iter(self,new_samples,new_evaluations):
        #FIXME : Haven't found a way to make it in n^2,m^2, whatever
        samples = torch.cat([self.samples,new_samples],dim=0)
        evaluations = torch.cat([self.evaluations,new_evaluations],dim=0)
        self.update_samples(samples,evaluations,
                            choose_inducing_points=False,
                            empirical_init=False)
        self.make_kernel_matrix()
        self.make_cholesky_factor()
        self.make_weights_vector()
        
    @property
    def lengthscale(self):
        return self._param_transform(self._raw_lengthscale)
    
    @property
    def outputscale(self):
        return self._param_transform(self._raw_outputscale)
    
    @property
    def mean_center(self):
        return self._raw_center
    
    @property
    def mean_lengthscale(self):
        return self._param_transform(self._raw_mean_lengthscale)
    
    @property
    def mean_constant(self):
        return self._raw_mean_constant
    
    @property
    def num_samples(self):
        return self.samples.shape[0]
    
    @property
    def noise(self):
        return self._param_transform(self._raw_noise)
        
    @noise.setter
    def noise(self,value):
        self._raw_noise = self._inv_param_transform(value)

    @property
    def Kuu(self):
        return self.Luu@self.Luu.t()
                
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

    def _choose_inducing_points(self,samples,n_inducing):
        _,self.inducing = utils.lloyd(self.samples,n_inducing)

    def _zvec(self,mu,cov,X,theta,l):
        C = cov + torch.diag(l**2)
        L = torch.cholesky(C,upper=False)
        Xm = X - mu #nxd#
        LX = torch.triangular_solve(Xm.transpose(1,0),L,upper=False)[0] #d x n
        expoent = -0.5*torch.sum(LX**2,dim=0) #(n,)
        det = torch.prod(1/l**2)*torch.prod(torch.diag(L))**2 #|I + A^-1B|
        zvec = theta/torch.sqrt(det)*torch.exp(expoent) #(n,)
        return zvec
    
    def _zvecdiag(self,mu,cov,X,theta,l):
        C = cov + l**2 #(d,)
        Xm = X - mu #(nxd)
        expoent = -0.5*torch.sum(Xm**2/C,dim=1)
        det = torch.prod(1/l**2*C)
        zvec = theta/torch.sqrt(det)*torch.exp(expoent) #(n,)
        return zvec
        
    def _zvecmvn(self,mu,cov,X,theta,l,weights):
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
        return zvec

    def _zvecmvndiag(self,mu,cov,X,theta,l,weights):
        # mu : (t,d)
        # cov : (t,d)
        # X : (n,d)
        C = (cov + l**2).unsqueeze(1) #(t,1,d)
        Xm = X.unsqueeze(0) - mu.unsqueeze(1) #(t,n,d)
        expoent = -0.5*torch.sum(Xm**2/C,dim=-1) #(t,n)
        det = torch.prod(1/l**2*C,dim=-1) #(t,1)
        vec_ = theta/torch.sqrt(det)*torch.exp(expoent) #(t,n)
        zvec = (weights.reshape(-1,1)*vec_).sum(dim=0) #(n,)
        return zvec
    
    def _vardetterm(self,cov,theta,l):
        Ccov = cov + 0.5*torch.diag(l**2)
        Lcov = torch.cholesky(Ccov,upper=False)
        detcov = torch.prod(2/(l**2))*torch.prod(torch.diag(Lcov))**2
        term = theta/torch.sqrt(detcov)
        return term

    def _vardettermdiag(self,cov,theta,l):
        Ccov = cov + 0.5*(l**2)
        detcov = torch.prod(2/(l**2)*Ccov)
        term = theta/torch.sqrt(detcov)
        return term

    def _new_variance(self,Xnew,mu=None,cov=None):
        #Uses current mu as reference
        if utils.is_none(mu) or utils.is_none(cov):
            if self._intvectype != "SINGLE":
                raise TypeError
            else:
                from_current_mucov = True
        else:
            from_current_mucov = False
        theta = self.outputscale
        l = self.lengthscale
        Xtilde = torch.cat([self.samples,Xnew],dim=0)
        Kuf = utils.rbf_kernel(self.inducing,Xtilde,
                                        self.outputscale,
                                        self.lengthscale)
        M = self.inducing.shape[0]
        N = Xtilde.shape[0]
        if from_current_mucov:
            zvecnew = self._zvec(mu,cov,self.inducing,theta,l)
            zvecstar = torch.cat([self.integralvector,zvecnew],dim=0) #(n+k,)
            varterm1 = self.vardetterm
        else:
            zvecstar = self._zvec(mu,cov,self.inducing,theta,l)
            varterm1 = self._vardetterm(cov,theta,l)
        #FIXME: REALLY DUMB AND INNEFICIENT WAY TO DO THIS
        W = torch.triangular_solve(Kuf,self.Luu,upper=False)[0].t() #(N,M) #W W^T = Qff
        Ddiag = self.noise*torch.ones(N) #(N,)
        Wd = (W/Ddiag.reshape(-1,1)) #(N,M)
        B = torch.eye(M) + torch.matmul(W.t(),Wd) #(M,M)
        LB = torch.cholesky(B,upper=False) #(M,M)
        # D^-1 - Wd Lb^-T Lb^-1 Wd^T = (Qff + D)^{-1}
        LBWdt = torch.triangular_solve(Wd.t(),LB,upper=False)[0] #(M,N)
        LuuKuf = torch.triangular_solve(Kuf,self.Luu,upper=False)[0] #(u,f)
        LuuKuz = torch.triangular_solve(zvecstar.reshape(-1,1),
                                        self.Luu,upper=False)[0] #(u,x)
        Qzf = torch.matmul(LuuKuz.t(),LuuKuf)
        varterm2 = torch.matmul(Qzf,Qzf.t())/self.noise #Substitute when FITC
        varterm3 = torch.matmul(torch.matmul(Qzf,LBWdt.t()),
                                   torch.matmul(LBWdt,Qzf.t()))
        var = varterm1 - (varterm2 - varterm3)
        return var

    def _persistent_names(self):
        names = ["samples","evaluations","inducing",
                 "jitter","Luu","Ddiag","Kuf","LBWdt",
                 "weights_vector","integral_vector",
                 "vardetterm","_raw_noise",
                 "_raw_outputscale","_raw_lengthscale",
                 "mean_center","_raw_mean_lengthscale","_raw_mean_constant"]
        return names

    def _unwarped_k_and_m(self,X):
        m_ = self._mean(X)
        K_ = self.kfunc(X,X,symmetrize=True)
        m = torch.exp(m_ + 0.5*torch.diag(K_).reshape(-1,1))
        K = (torch.exp(K_)-1)*(m@m.t())
        return m,K

    def _unwarped_k_2(self,X,Y):
        mx_,Kx_ = self._mean(X),self.kfunc(X,X,diag=True).reshape(-1,1)
        my_,Ky_ = self._mean(Y),self.kfunc(Y,Y,diag=True).reshape(-1,1)
        mx = torch.exp(mx_ + 0.5*Kx_.reshape(-1,1))
        my = torch.exp(my_ + 0.5*Ky_.reshape(-1,1))
        K_ = self.kfunc(X,Y)
        K = (torch.exp(K_)-1)*(mx@my.t())
        return K

    def kfunc(self,X,Y,symmetrize=False,diag=False):
        return utils.rbf_kernel(X,Y,self.outputscale,
                                self.lengthscale,symmetrize=symmetrize,
                                diag=diag)

    #change device
    def change_device(self,device):
        for name in self._persistent_names():
            if hasattr(self,name) and torch.is_tensor(getattr(self,name)):
                setattr(self,name,getattr(self,name).to(device))
        self.device_type = "device"
    
class BMC_SQM(_BMCSparseBase):
    def __init__(self,dim,minvalue=1e-6,
                 theta0=1.0,l0=1.0,
                 center0=0.0,lmeans0=10.0,constant0=0.0,
                 jitter=1e-4,noise=1e-4,device="cpu",**kwargs):
        """
            dim : dimension of the distribution
        """
        super().__init__(dim,minvalue,jitter=jitter,noise=noise,device=device)
        self._set_init_params(theta0,l0,center0,lmeans0,constant0,device=device)        
    
    @property
    def mean_center(self):
        return self._raw_center
    
    @mean_center.setter
    def mean_center(self,value):
        self._raw_center = value
        
    @property
    def mean_lengthscale(self):
        return self._param_transform(self._raw_mean_lengthscale)
    
    @property
    def mean_constant(self):
        return self._raw_mean_constant
    
    def _set_init_params(self,theta0,l0,center0,lmeans0,constant0,
                         device="cpu"):
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
                self.mean_center,self._raw_mean_lengthscale,self._raw_mean_constant]
    
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


class BMC_SCM(_BMCSparseBase):
    def __init__(self,dim,minvalue=1e-6,
                 theta0=1.0,l0=1.0,constant0=0.0,
                 jitter=1e-4,noise=1e-4,device="cpu",**kwargs):
        """
            dim : dimension of the distribution
        """
        super().__init__(dim,minvalue,jitter=jitter,noise=noise,device=device)
        self._set_init_params(theta0,l0,constant0,device=device)        
        
    @property
    def mean_constant(self):
        return self._raw_mean_constant
    
    def _set_init_params(self,theta0,l0,constant0,device="cpu"):
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
        self._raw_mean_constant = torch.tensor(constant0,device=device)

    def _set_init_params_empirically(self,device="cpu"):
        if device != "cpu": 
            raise NotImplementedError
        assert hasattr(self,"samples") and hasattr(self,"evaluations")
        self._raw_mean_consatnt = torch.mean(self.evaluations)
        raw_mean_constant,raw_center_index = torch.max(self.evaluations,dim=0)
        outputscale = torch.std(self.evaluations - self._mean(self.samples))**2
        self._raw_outputscale = self._inv_param_transform(outputscale)
        lengthscale = torch.std(self.samples,dim=0)
        self._raw_lengthscale = self._inv_param_transform(lengthscale)
    
    def _params(self):
        return [self._raw_outputscale,self._raw_lengthscale,self._raw_mean_constant]
    
    def _mean(self,x):
        return self.mean_constant

    def _get_mean_factor(self,mu,cov,diag=False):
        mean_factor = self.mean_constant
        return mean_factor
    
    def _get_mean_factor_mixture(self,mu,cov,weights,diag=False):
        mean_factor = self.mean_constant      
        return mean_factor
    

class BMC_SFM(_BMCSparseBase):
    def __init__(self,dim,minvalue=1e-6,
                 theta0=1.0,l0=1.0,constant0=0.0,
                 jitter=1e-4,noise=1e-4,device="cpu",**kwargs):
        """
            dim : dimension of the distribution
            The schema here (experimental) is that if constant is -math.inf,
            we get a very low constant, min_eval - D*(max_val - min_eval)
        """
        super().__init__(dim,minvalue,jitter=jitter,noise=noise)
        self._set_init_params(theta0,l0,constant0,device=device)        
        
    @property
    def mean_constant(self):
        if self._raw_mean_constant != -math.inf:
            return self._raw_mean_constant
        else:
            eval_max = max(self.evaluations)
            eval_min = min(self.evaluations)
            D = 5
            return eval_min - D*(eval_max - eval_min)

    @mean_constant.setter
    def mean_constant(self,value):
        if value != -math.inf:
            self._raw_mean_constant = torch.tensor(value)
        else:
            self._raw_mean_constant = -math.inf

    def _set_init_params(self,theta0,l0,constant0,device="cpu"):
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
    
    def _set_init_params_empirically(self,device="cpu"):
        if device != "cpu": 
            raise NotImplementedError
        assert hasattr(self,"samples") and hasattr(self,"evaluations")
        outputscale = torch.std(self.evaluations - self._mean(self.samples))**2
        self._raw_outputscale = self._inv_param_transform(outputscale)
        lengthscale = torch.std(self.samples,dim=0)
        self._raw_lengthscale = self._inv_param_transform(lengthscale)
    
    def _params(self):
        return [self._raw_outputscale,self._raw_lengthscale]
        
    def _mean(self,x):
        return self.mean_constant*torch.ones((x.shape[0],1),
                                             device=x.device)

    def _get_mean_factor(self,mu,cov,diag=False):
        mean_factor = self.mean_constant
        return mean_factor
    
    def _get_mean_factor_mixture(self,mu,cov,weights,diag=False):
        mean_factor = self.mean_constant      
        return mean_factor
    