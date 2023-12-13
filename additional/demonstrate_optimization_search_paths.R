### compare the optimization path for the gradient descent method
### and the newton method (with linesearch)

### use the in-class weight example which uses synthetic data

library(tidyverse)

### the gradient and Hessian will be calculated numerically
### if you want to run this script you must download and install
### the numDeriv package, the RStudio packages installer can help
library(numDeriv)

### specify the true parameters
mu_true <- 254.5
sigma_true <- 2.5

### generate random observations
set.seed(5001)
x <- rnorm(n = 50, mean = mu_true, sd = sigma_true)

### visualize the random observations as a run chart
tibble::tibble(x = x) %>% 
  tibble::rowid_to_column() %>% 
  ggplot(mapping = aes(x = rowid, y = x)) +
  geom_hline(yintercept = mu_true, color = "red", size = 1.1) +
  geom_hline(yintercept = c(mu_true - 2*sigma_true, mu_true + 2*sigma_true),
             color = "red", linetype = "dashed", size = 1.1) +
  geom_line(size = 1.15, color = "grey50") +
  geom_point(size = 4.5, color = "black") +
  labs(x = "observation index") +
  theme_bw()

### work with just the first 10 measurements

### define the list of required information

num_obs <- 10

info_use <- list(
  xobs = x[0:num_obs],
  mu_0 = 250,
  tau_0 = 2,
  sigma_lwr = 0.5,
  sigma_upr = 5.5
)

### run style chart for the first 10 measurements
tibble::tibble(x = info_use$xobs) %>% 
  tibble::rowid_to_column() %>% 
  ggplot(mapping = aes(x = rowid, y = x)) +
  geom_line(size = 1.25, color = "grey50") +
  geom_point(size = 7.5, color = "black") +
  labs(x = "observation index") +
  theme_bw()

### define log-posterior function with gaussian likelihood
### gaussian prior on mu and uniform prior on sigma
### phi is the logit-transformed sigma parameter
my_logpost <- function(theta, my_info)
{
  # unpack
  lik_mu <- theta[1]
  lik_phi <- theta[2]
  
  # backtransform
  lik_sigma <- my_info$sigma_lwr + 
    (my_info$sigma_upr - my_info$sigma_lwr) * boot::inv.logit(lik_phi)
  
  # log-liklhood
  log_lik <- sum(dnorm(x = my_info$xobs,
                       mean = lik_mu,
                       sd = lik_sigma,
                       log = TRUE))
  
  # log-priors
  log_prior_mu <- dnorm(x = lik_mu, 
                        mean = my_info$mu_0,
                        sd = my_info$tau_0,
                        log = TRUE)
  
  log_prior_sigma <- dunif(x = lik_sigma,
                           min = my_info$sigma_lwr,
                           max = my_info$sigma_upr,
                           log = TRUE)
  
  log_prior <- log_prior_mu + log_prior_sigma
  
  # log-derivative adjustment
  log_deriv_adjust <- log(my_info$sigma_upr - my_info$sigma_lwr) + 
    log(boot::inv.logit(lik_phi)) + 
    log(1 - boot::inv.logit(lik_phi))
  
  # sum together
  log_lik + log_prior + log_deriv_adjust
}

### define a grid parmaeter values 
param_grid <- expand.grid(lik_mu = seq(info_use$mu_0 - 3*info_use$tau_0,
                                       info_use$mu_0 + 3*info_use$tau_0,
                                       length.out = 251),
                          lik_phi = seq(boot::logit(1e-2),
                                        boot::logit(0.99),
                                        length.out = 251),
                          KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE) %>% 
  as.data.frame() %>% tibble::as_tibble()

### visualize the log-posterior surface over the parameter grid
param_grid %>% 
  rowwise() %>% 
  mutate(log_post = my_logpost(c(lik_mu, lik_phi), info_use)) %>% 
  ungroup() %>% 
  mutate(log_post_2 = log_post - max(log_post)) %>% 
  ggplot(mapping = aes(x = lik_mu, y = lik_phi)) +
  geom_raster(mapping = aes(fill = log_post_2)) +
  stat_contour(mapping = aes(z = log_post_2),
               breaks = log(c(0.01/100, 0.01, 0.1, 0.5, 0.9)),
               size = 2.2,
               color = "black") +
  scale_fill_viridis_c(guide = 'none', 
                       option = "viridis",
                       limits = log(c(0.01/100, 1.0))) +
  labs(x = expression(mu), y = expression(varphi)) +
  theme_bw()

### refine the parameter grid around the posterior mode

param_grid_refine <- expand.grid(lik_mu = seq(248, 260, length.out = 251),
                                 lik_phi = seq(boot::logit(0.05),
                                               boot::logit(0.99), 
                                               length.out = 251),
                                 KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE) %>% 
  as.data.frame() %>% tibble::as_tibble()

### visualize the posterior surface with the refined grid
log_post_refine_df <- param_grid_refine %>% 
  rowwise() %>% 
  mutate(log_post = my_logpost(c(lik_mu, lik_phi), info_use)) %>% 
  ungroup()

log_post_refine_df %>% 
  mutate(log_post_2 = log_post - max(log_post)) %>% 
  ggplot(mapping = aes(x = lik_mu, y = lik_phi)) +
  geom_raster(mapping = aes(fill = log_post_2)) +
  stat_contour(mapping = aes(z = log_post_2),
               breaks = log(c(0.01/100, 0.01, 0.1, 0.5, 0.9)),
               size = 2.2,
               color = "black") +
  scale_fill_viridis_c(guide = 'none', 
                       option = "viridis",
                       limits = log(c(0.01/100, 1.0))) +
  labs(x = expression(mu), y = expression(varphi)) +
  theme_bw()

### --- setup functions to support performing optimization --- ###

### for optimization work with the negative log-posterior
### work simple, hardcode the logposterior function
negative_logpost <- function(theta, logpost_info)
{
  -my_logpost(theta, logpost_info)
}

### check the evaluation of the gradient at one point
numDeriv::grad(negative_logpost, c(250, -1), logpost_info = info_use)

### check the evaluation of the Hessian matrix at one point
numDeriv::hessian(negative_logpost, c(250, -1), 
                  method="Richardson", method.args=list(), 
                  info_use)

### define a function which calculates the gradient descent step
grad_descent_step <- function(theta, logpost_info, step_size = 0.001)
{
  # calculate gradient
  g <- numDeriv::grad(negative_logpost, theta, 
                      method="Richardson", side=NULL, method.args=list(),
                      logpost_info)
  
  # direction
  p <- -g
  
  # take the step with the applied learning rate
  theta + step_size * as.numeric(p)
}

### define a function which calculates the full newton step
newton_step <- function(theta, logpost_info, step_size = 1)
{
  # calculate gradient
  g <- numDeriv::grad(negative_logpost, theta, 
                      method="Richardson", side=NULL, method.args=list(),
                      logpost_info)
  
  # calculate Hessian matrix
  Hmat <- numDeriv::hessian(negative_logpost, theta, 
                            method="Richardson", method.args=list(), 
                            logpost_info)
  
  # update
  as.numeric(theta - step_size * as.numeric(solve(Hmat, as.matrix(g))))
}

### wrapper function to the gradient evaluation which allows trying out
### several different learning rates at the same guess

wrap_grad_step <- function(step_size, theta, logpost_info, grad_func_use,
                           var_names)
{
  x <- grad_func_use( theta, logpost_info, step_size)
  
  data.frame( t(x) ) %>% tibble::as_tibble() %>% 
    purrr::set_names(var_names) %>% 
    mutate(step_size = step_size)
}

### define a function which shows the initial guess on the surface
viz_init_guess_only <- function(init_theta_df)
{
  gg <- log_post_refine_df %>% 
    mutate(log_post_2 = log_post - max(log_post)) %>% 
    ggplot(mapping = aes(x = lik_mu, y = lik_phi)) +
    geom_raster(mapping = aes(fill = log_post_2)) +
    stat_contour(mapping = aes(z = log_post_2),
                 breaks = log(c(0.01/100, 0.01, 0.1, 0.5, 0.9)),
                 size = 2.2,
                 color = "black") +
    geom_point(data = init_theta_df,
               size = 13, shape = 22, alpha = 1.0,
               fill = 'white', color = 'red') +
    geom_text(data = init_theta_df,
              mapping = aes(label = guess_id),
              color = "black", size = 8) +
    scale_fill_viridis_c(guide = 'none', option = "viridis",
                         limits = log(c(0.01/100, 1.0))) +
    labs(x = expression(mu), y = expression(varphi)) +
    theme_bw()
  
  print( gg )
}

### define a list of initial guess values
init_guess_list <- list(
  c(250, 2),
  c(257, -1),
  c(253, 0.5),
  c(250, -1.5),
  c(253, 2.75),
  c(258, 2.9),
  c(252.1, -2.6),
  c(250.7, 3.75),
  c(259, -2.5)
)

### convert the list into a tibble
init_guess_df <- purrr::map_dfr(init_guess_list,
                                function(avec){as.data.frame(t(avec)) %>% 
                                    purrr::set_names(c("lik_mu", "lik_phi"))}) %>% 
  tibble::rowid_to_column("guess_id")

### visualize the initial guess values on the surface
viz_init_guess_only(init_guess_df)

### use several learning rates to visualize the search path direction
### directly compare the gradient descent direction to the newton direction

viz_compare_directions <- function(a_theta, guess_id, grad_desc_lr, newton_lr)
{
  # calculate updates using gradient descent and the provided
  # learning rates
  grad_desc_at <- purrr::map_dfr(grad_desc_lr,
                                 wrap_grad_step,
                                 theta = a_theta,
                                 logpost_info = info_use,
                                 grad_func_use = grad_descent_step,
                                 var_names = c("lik_mu", "lik_phi"))
  
  # calculate updates using newton and the provided learning rates
  newton_at <- purrr::map_dfr(newton_lr,
                              wrap_grad_step,
                              theta = a_theta,
                              logpost_info = info_use,
                              grad_func_use = newton_step,
                              var_names = c("lik_mu", "lik_phi"))
  
  # draw an arrow for each respective search path direction
  gg <- log_post_refine_df %>% 
    mutate(log_post_2 = log_post - max(log_post)) %>% 
    ggplot(mapping = aes(x = lik_mu, y = lik_phi)) +
    geom_raster(mapping = aes(fill = log_post_2)) +
    stat_contour(mapping = aes(z = log_post_2),
                 breaks = log(c(0.01/100, 0.01, 0.1, 0.5, 0.9)),
                 size = 2.2,
                 color = "black") +
    geom_point(data = tibble::tibble(lik_mu = a_theta[1],
                                     lik_phi = a_theta[2]),
               size = 13, shape = 22, alpha = 1.0,
               fill = 'grey', color = 'red') +
    geom_text(data = tibble::tibble(lik_mu = a_theta[1],
                                    lik_phi = a_theta[2],
                                    guess_id = guess_id),
              mapping = aes(label = guess_id), 
              size = 8, color = 'black') +
    geom_path(data = grad_desc_at,
              mapping = aes(x = lik_mu, y = lik_phi,
                            color = 'Gradient Descent'),
              size= 1.,
              arrow = arrow(type='closed')) +
    geom_path(data = newton_at,
              mapping = aes(x = lik_mu, y = lik_phi,
                            color = "Newton Method"),
              size = 1.1,
              arrow = arrow(type='closed')) +
    scale_fill_viridis_c(guide = 'none', option = "viridis",
                         limits = log(c(0.01/100, 1.0))) +
    scale_color_manual("", 
                       values = c("Gradient Descent" = "cyan",
                                  "Newton Method" = "magenta")) +
    labs(x = expression(mu), y = expression(varphi)) +
    theme_bw() +
    theme(legend.position = "top",
          legend.key = element_rect(fill = "grey"))
  
  print( gg )
}

### compare the search path directions at the initial guess for the
### first initial guess
viz_compare_directions(init_guess_list[[1]], init_guess_df$guess_id[[1]],
                       grad_desc_lr = exp(seq(-7, -1, length.out = 3)),
                       newton_lr = exp(seq(-7, 0, length.out = 3)))

### compare the search path directions for the second initial guess
viz_compare_directions(init_guess_list[[2]], init_guess_df$guess_id[[2]],
                       grad_desc_lr = exp(seq(-7, -3, length.out = 3)),
                       newton_lr = exp(seq(-7, 0, length.out = 3)))

### compare the search path directions for the third initial guess
viz_compare_directions(init_guess_list[[3]], init_guess_df$guess_id[[3]],
                       grad_desc_lr = exp(seq(-7, -0.5, length.out = 3)),
                       newton_lr = exp(seq(-7, 0, length.out = 3)))

### 4th initial guess
viz_compare_directions(init_guess_list[[4]], init_guess_df$guess_id[[4]],
                       grad_desc_lr = exp(seq(-7, -5, length.out = 3)),
                       newton_lr = exp(seq(-7, 0, length.out = 3)))

### 5th initial guess -- this is a BAD initial guess...gives problems
### for the NEWTON method!!!!!!
viz_compare_directions(init_guess_list[[5]], init_guess_df$guess_id[[5]],
                       grad_desc_lr = exp(seq(-7, -0.5, length.out = 3)),
                       newton_lr = exp(seq(-7, -1, length.out = 3)))

### 6th initial guess...again aa BAD initial guess!!!!
viz_compare_directions(init_guess_list[[6]], init_guess_df$guess_id[[6]],
                       grad_desc_lr = exp(seq(-7, -1, length.out = 3)),
                       newton_lr = exp(seq(-7, -2, length.out = 3)))

### 7th initial guess...NOT a BAD initial guess
viz_compare_directions(init_guess_list[[7]], init_guess_df$guess_id[[7]],
                       grad_desc_lr = exp(seq(-7, -5, length.out = 3)),
                       newton_lr = exp(seq(-7, 0, length.out = 3)))

### 8th initial guess
viz_compare_directions(init_guess_list[[8]], init_guess_df$guess_id[[8]],
                       grad_desc_lr = exp(seq(-7, -0.5, length.out = 3)),
                       newton_lr = exp(seq(-7, -3, length.out = 3)))

### 9th initial guess
viz_compare_directions(init_guess_list[[9]], init_guess_df$guess_id[[9]],
                       grad_desc_lr = exp(seq(-7, -5, length.out = 3)),
                       newton_lr = exp(seq(-7, 0, length.out = 3)))

### define a function for the gradient descent update
### which returns the update values and the gradient
### allows monitoring the gradient with the result

make_grad_desc <- function(theta, logpost_info, step_size = 0.001)
{
  # calculate gradient
  g <- numDeriv::grad(negative_logpost, theta, 
                      method="Richardson", side=NULL, method.args=list(),
                      logpost_info)
  
  # direction
  p <- -g
  
  # take the step with the applied learning rate
  tnew <- theta + step_size * as.numeric(p)
  
  list(theta = tnew, g = g)
}


### since some initial guesses can be BAD...need to use Newton method
### with backtracking line search to identify the OPTIMAL learning rate
### at a given point. 
###
### IF the learning rate descrases below a HARD CODED
### threshold simply use a gradient descent step there...

make_newton_linesearch <- function(theta, logpost_info, a = 0.01, b = 0.8)
{
  # negative log-posterior
  nlpvalue <- negative_logpost(theta, logpost_info)
  
  # calculate gradient vector
  g <- numDeriv::grad(negative_logpost, theta,
                      method = "Richardson", side=NULL, method.args=list(),
                      logpost_info)
  
  # calculate hessian matrix
  Hmat <- numDeriv::hessian(negative_logpost, theta,
                            method = "Richardson", method.args=list(),
                            logpost_info)
  
  # search direction
  p <- -solve(Hmat + 1e-8 * diag(length(g)), as.matrix(g))
  
  # back tracking line search
  step_size <- 1 # full newton
  u <- a * as.numeric(t(as.matrix(g)) %*% p)
  
  ### just hard code the max number of line search backtracks
  for(nt in 1:201){
    ff1 <- negative_logpost(theta + step_size * as.numeric(p), logpost_info)
    
    ff2 <- nlpvalue + step_size * u
    
    if( ff1 <= ff2 ){
      break
    } else {
      step_size <- b * step_size
    }
  }
  
  if(step_size < 1e-10){
    # browser()
    # step size will be incredibly small, just use a scaled gradient descent
    step_size <- 0.005
    p <- -g
  }
  
  # take the damped newton step
  tnew <- theta + step_size * as.numeric(p)
  
  list(theta = tnew, g = g)
}

### create a wrapper function which runs the optimization
### from an initial guess for a given number of steps

### for simplicity...HARD CODE a break point of the
### l2norm of the gradient at 1e-4

run_optimize <- function(start_value, num_steps, grad_func, logpost_info, ...)
{
  # initialize
  res <- vector(mode = 'list', length = num_steps + 1)
  gres <- vector(mode = 'list', length = num_steps + 1)
  
  # browser()
  
  res[[1]] <- start_value
  
  for(k in 2:(num_steps+1)){
    temp_res <- grad_func(res[[k - 1]], logpost_info, ...)
    
    res[[k]] <- temp_res$theta
    gres[[k - 1]] <- temp_res$g
    
    if( sqrt( sum( temp_res$g^2 ) ) < 1e-4 ) { break }
  }
  
  list(theta_list = res, grad_list = gres)
}


### unpack the iteration results

tidy_opt_results <- function(t_vec, g_vec, iter_id)
{
  list(lik_mu = t_vec[1],
       lik_phi = t_vec[2],
       grad_l2_norm = sqrt( sum( g_vec^2 ) ),
       grad_linf_norm = max(g_vec),
       iter_id = iter_id)
}

### define a function which executes the optimization and then the
### tidying of the results

run_tidy_optimize <- function(start_value, num_steps, grad_func, logpost_info, ...)
{
  res_list <- run_optimize(start_value, num_steps, grad_func, logpost_info, ...)
  
  purrr::pmap_dfr(list( res_list$theta_list,
                        res_list$grad_list,
                        seq_along(res_list$theta_list) - 1),
                  tidy_opt_results)
}

### iterate over the initial guess values, use a large number for the
### max number of steps, hard code the step size to be 0.005
grad_desc_opt_results_all <- purrr::map(init_guess_list,
                                        run_tidy_optimize,
                                        num_steps = 5001,
                                        grad_func = make_grad_desc,
                                        logpost_info = info_use,
                                        step_size = 0.005)

### plot the l2-norm of the gradient which is the convergence monitor
purrr::map2_dfr(grad_desc_opt_results_all, 
                seq_along(grad_desc_opt_results_all),
                function(ll, id){ll %>% mutate(guess_id = id)}) %>% 
  ggplot(mapping = aes(x = iter_id, y = grad_l2_norm)) +
  geom_line(mapping = aes(group = interaction(guess_id)),
            size = 1.1) +
  facet_wrap(~guess_id, labeller = "label_both") +
  theme_bw()

### zoom in on the scale and plot all guesses together
purrr::map2_dfr(grad_desc_opt_results_all, 
                seq_along(grad_desc_opt_results_all),
                function(ll, id){ll %>% mutate(guess_id = id)}) %>% 
  mutate(opt_type = "Gradient Descent") %>% 
  ggplot(mapping = aes(x = iter_id, y = grad_l2_norm)) +
  geom_line(mapping = aes(group = interaction(guess_id),
                          color = as.factor(guess_id)),
            size = 1.1) +
  facet_wrap(~opt_type) +
  coord_cartesian(ylim = c(-0.1, 2)) +
  scale_color_viridis_d("Guess") +
  theme_bw()

### what if we used a smaller step size?
grad_desc_opt_results_small_step <- purrr::map(init_guess_list,
                                               run_tidy_optimize,
                                               num_steps = 10001,
                                               grad_func = make_grad_desc,
                                               logpost_info = info_use,
                                               step_size = 0.001)

### plot the l2norm of the gradient for the small step size
purrr::map2_dfr(grad_desc_opt_results_small_step, 
                seq_along(grad_desc_opt_results_small_step),
                function(ll, id){ll %>% mutate(guess_id = id)}) %>% 
  mutate(opt_type = "Gradient Descent small learning rate") %>% 
  ggplot(mapping = aes(x = iter_id, y = grad_l2_norm)) +
  geom_line(mapping = aes(group = interaction(guess_id),
                          color = as.factor(guess_id)),
            size = 1.1) +
  facet_wrap(~opt_type) +
  coord_cartesian(ylim = c(-0.1, 2)) +
  scale_color_viridis_d("Guess") +
  theme_bw()

### use the newton method with backtracking line search to handle
### the bad initial guesses
newton_opt_results_all <- purrr::map(init_guess_list,
                                     run_tidy_optimize,
                                     num_steps = 5001,
                                     grad_func = make_newton_linesearch,
                                     logpost_info = info_use,
                                     a = 0.01, b = 0.8)

### plot the l2norm of the gradient with respect to the number of iterations
purrr::map2_dfr(newton_opt_results_all, 
                seq_along(newton_opt_results_all),
                function(ll, id){ll %>% mutate(guess_id = id)}) %>% 
  ggplot(mapping = aes(x = iter_id, y = grad_l2_norm)) +
  geom_line(mapping = aes(group = interaction(guess_id)),
            size = 1.1) +
  facet_wrap(~guess_id, labeller = "label_both") +
  theme_bw()

### only the BAD initial guess values require a large number of iterations!!!
### let each facet have separate scales
purrr::map2_dfr(newton_opt_results_all, 
                seq_along(newton_opt_results_all),
                function(ll, id){ll %>% mutate(guess_id = id)}) %>% 
  ggplot(mapping = aes(x = iter_id, y = grad_l2_norm)) +
  geom_line(mapping = aes(group = interaction(guess_id)),
            size = 1.1) +
  facet_wrap(~guess_id, labeller = "label_both", scales = "free") +
  theme_bw()

### zoom in on the iteration - xaxis - scale and the l2-norm - yaxis - scale
### only the BAD initial guess values require more than 20 iterations to 
### converge for this example!!!
purrr::map2_dfr(newton_opt_results_all, 
                seq_along(newton_opt_results_all),
                function(ll, id){ll %>% mutate(guess_id = id)}) %>% 
  ggplot(mapping = aes(x = iter_id, y = grad_l2_norm)) +
  geom_line(mapping = aes(group = interaction(guess_id)),
            size = 1.1) +
  coord_cartesian(xlim = c(0, 21), ylim = c(0, 10)) +
  facet_wrap(~guess_id, labeller = "label_both") +
  theme_bw()

### look at the two BAD starting guess values can see where it
### transitions from gradient descent to the newton method
purrr::map2_dfr(newton_opt_results_all, 
                seq_along(newton_opt_results_all),
                function(ll, id){ll %>% mutate(guess_id = id)}) %>% 
  filter(guess_id %in% c(5, 6)) %>% 
  ggplot(mapping = aes(x = iter_id, y = grad_l2_norm)) +
  geom_line(mapping = aes(group = interaction(guess_id)),
            size = 1.1) +
  coord_cartesian(xlim = c(0, 301)) +
  facet_wrap(~guess_id, labeller = "label_both") +
  theme_bw()



### directly compare the optimization paths for gradient descent
### and the Newton method for each initial guess

viz_compare_opt_results <- function(graddesc_df, newton_df, guess_id)
{
  init_values <- graddesc_df %>% slice(1)
  
  gg <- log_post_refine_df %>% 
    mutate(log_post_2 = log_post - max(log_post)) %>% 
    ggplot(mapping = aes(x = lik_mu, y = lik_phi)) +
    geom_raster(mapping = aes(fill = log_post_2)) +
    stat_contour(mapping = aes(z = log_post_2),
                 breaks = log(c(0.01/100, 0.01, 0.1, 0.5, 0.9)),
                 size = 2.2,
                 color = "black") +
    geom_point(data = init_values,
               size = 13, shape = 22, alpha = 1.0,
               fill = 'grey', color = 'red') +
    geom_text(data = init_values %>% mutate(guess_id = guess_id),
              mapping = aes(label = guess_id),
              color = 'black', size = 8) +
    geom_path(data = graddesc_df,
              mapping = aes(x = lik_mu, y = lik_phi,
                            color = 'Gradient Descent'),
              size= 1.0) +
    geom_path(data = newton_df,
              mapping = aes(x = lik_mu, y = lik_phi,
                            color = "Newton Method"),
              size= 1.0) +
    geom_point(data = graddesc_df,
               mapping = aes(x = lik_mu, y = lik_phi,
                             color = 'Gradient Descent')) +
    geom_point(data = newton_df,
               mapping = aes(x = lik_mu, y = lik_phi,
                             color = 'Newton Method'),
               size = 2.21) +
    scale_fill_viridis_c(guide = 'none', 
                         option = "viridis",
                         limits = log(c(0.01/100, 1.0))) +
    scale_color_manual("", 
                       values = c("Gradient Descent" = "cyan",
                                  "Newton Method" = "magenta")) +
    labs(x = expression(mu), y = expression(varphi)) +
    theme_bw() +
    theme(legend.position = "top",
          legend.key = element_rect(fill = "grey"))
  
  print( gg )
}

purrr::pwalk(list(grad_desc_opt_results_all, 
                  newton_opt_results_all,
                  seq_along(init_guess_list)),
             viz_compare_opt_results)
