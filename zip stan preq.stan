data {
  int<lower=0> n_pa; //number of observations
  matrix[n_pa,9] x_pa; // matrix with covariates
  int y_pa[n_pa]; // random effect
  int O_pa[n_pa];//observations
  matrix[9,9] inv_xtx_pa;// of matrix with covariates to use in g-prior
  
}

transformed data{
  real x1[n_pa];
  real x2[n_pa];
  real x3[n_pa];
  real x4[n_pa];
  real x5[n_pa];
  real x6[n_pa];
  real sp[n_pa];
  real su[n_pa];
  real fa[n_pa];
  
  for (i in 1:n_pa){
    x1[i]=x_pa[i,1];
    x2[i]=x_pa[i,2];
    x3[i]=x_pa[i,3];
    x4[i]=x_pa[i,4];
    x5[i]=x_pa[i,5];
    x6[i]=x_pa[i,6];
    sp[i]=x_pa[i,7];
    su[i]=x_pa[i,8];
    fa[i]=x_pa[i,9];
    }
}  
parameters{
  
  vector[9] b;
  vector[9] c;
  real s;
  real random[5];
  real<lower=0,upper=1> K;
  real theta;
  real tau_btw;
  real gam;
  real mu1[n_pa];
}
transformed parameters{
  real lambda[n_pa];
  real M[n_pa];
  real C[n_pa];
  real U;
  real lamda;
  real phi;
  real sbtw;
  cov_matrix[9] prior_T ; //prior variance matrix used in g-prior
  real<lower=0,upper=1> p[n_pa];


  
  
  
  lamda=exp(s);
  prior_T=inv_xtx_pa*((n_pa*K)/(1-K))/lamda;
  sbtw=1/tau_btw;
  phi=exp(theta);
  U=(2*phi)/(1-exp(-2*phi));
  lambda[1] =exp( s + b[1]*x1[1] +b[2]*x2[1] +b[3]*x3[1] +b[4]*x4[1] +b[5]*x5[1] +b[6]*x6[1] +b[7]*sp[1]+ b[8]*su[1] +b[9]*fa[1] + random[y_pa[1]]);
  p[1]=inv_logit(s + c[1]*x1[1] +c[2]*x2[1] +c[3]*x3[1] +c[4]*x4[1] +c[5]*x5[1] +c[6]*x6[1] +c[7]*sp[1] +  c[8]*su[1] +c[9]*fa[1]) ;
  for(i in 2:n_pa){
    lambda[i] =exp(mu1[i]);
    C[i]= s + b[1]*x1[i] + b[2]*x2[i] +b[3]*x3[i]+b[4]*x4[i] +b[5]*x5[i] +b[6]*x6[i] +b[7]*sp[i]+b[8]*su[i] +b[9]*fa[i] + random[y_pa[i]] +  gam*O_pa[i-1];
    M[i]=C[i] + (lambda[i-1]- C[i])*exp(-phi);
    p[i]= inv_logit(s+c[1]*x1[i]+c[2]*x2[i]+c[3]*x3[i]+c[4]*x4[i]+c[5]*x5[i]+c[6]*x6[i]+c[7]*sp[i] +c[8]*su[i] +c[9]*fa[i]);
    
  }
    

}



model{
  
  vector[9] mu_beta;
  vector[9] mu_c;
  if(O_pa[1]==0)
   target += log_sum_exp(bernoulli_lpmf(1 | p[1]),bernoulli_lpmf(0 | p[1])+ poisson_lpmf(O_pa[1] | lambda[1]));
  else
   target += bernoulli_lpmf(0 | p[1])+ poisson_lpmf(O_pa[1] | lambda[1]);
  for(i in 2:n_pa){
    if(O_pa[i]==0)
      target += log_sum_exp(bernoulli_lpmf(1 | p[i]),bernoulli_lpmf(0 | p[i])+ poisson_lpmf(O_pa[i] | lambda[i]));
    else 
      target += bernoulli_lpmf(0 | p[i])+ poisson_lpmf(O_pa[i] | lambda[i]);
              
            }

  
  
  
  
  
  
  
  
  
  
  
  for (j in 1:9){
    mu_beta[j]=0.0;
    mu_c[j]=0.0;
    }
  for(i in 2:n_pa){  
  mu1[i]~normal(M[i],1/U);
  }
  K~beta(1,1);
  b[1:9]~multi_normal(mu_beta, prior_T);
  c[1:9]~multi_normal(mu_c, prior_T);
  O_pa~poisson(lambda);
   
  s ~ normal( 0, 1000);
  random~ normal(0, 1/tau_btw);
  gam ~ normal(0, 100);
  theta~normal(0,100);
  tau_btw ~gamma(0.01,0.01);
  
}
