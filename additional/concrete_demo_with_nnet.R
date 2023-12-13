### use the concrete regression problem example

library(tidyverse)

### load in the data from the modeldata package

data("concrete", package = "modeldata")

concrete %>% glimpse()

### check the names of the variables
concrete %>% names()

### as described earlier in the semester in the complete
### concrete regression demo, simplify this problem slightly
### by grouping the replications and thus focus just on the 
### AVERAGE strength
my_concrete <- concrete %>% 
  group_by(across(cement:age)) %>% 
  summarise(compressive_strength = mean(compressive_strength),
            .groups = 'drop')


my_concrete %>% glimpse()

### plot the response vs the inputs
my_concrete %>% 
  tibble::rowid_to_column("obs_id") %>% 
  pivot_longer(!c("obs_id", "compressive_strength")) %>% 
  ggplot(mapping = aes(x = value, y = compressive_strength)) +
  geom_point(alpha = 0.5) +
  facet_wrap(~name, scales = "free_x") +
  theme_bw()

### standardize all inputs and the response
ready_df <- my_concrete %>% 
  scale(center = TRUE, scale = TRUE) %>% 
  as.data.frame() %>% tibble::as_tibble()

ready_df %>% 
  tibble::rowid_to_column() %>% 
  pivot_longer(!c("rowid")) %>% 
  ggplot(mapping = aes(y = name, x = value)) +
  geom_boxplot(fill = 'grey') +
  theme_bw()

### fit with neuralnet

library(neuralnet)

set.seed(41231)
mod_1layer_small <- neuralnet(compressive_strength ~ .,
                              data = ready_df,
                              hidden = 3,
                              err.fct = 'sse',
                              act.fct = 'logistic',
                              linear.output = TRUE,
                              likelihood = TRUE)

plot(mod_1layer_small, rep = 'best', show.weights = FALSE)

mod_1layer_small$result.matrix %>% as.data.frame()

mod_1layer_small$weights

### now let's use more hidden units!

set.seed(41231)
mod_1layer_small_b <- neuralnet(compressive_strength ~ .,
                              data = ready_df,
                              hidden = 7,
                              err.fct = 'sse',
                              act.fct = 'logistic',
                              linear.output = TRUE,
                              likelihood = TRUE)

plot(mod_1layer_small_b, rep = 'best', show.weights = FALSE)

mod_1layer_small_b$result.matrix %>% as.data.frame()

mod_1layer_small_b$weights

### but it's usually better to try more hidden units than inputs!
### we are expanding the feature space!

set.seed(41231)
mod_1layer_medium <- neuralnet(compressive_strength ~ .,
                               data = ready_df,
                               hidden = 16,
                               err.fct = 'sse',
                               act.fct = 'logistic',
                               linear.output = TRUE,
                               likelihood = TRUE)

plot(mod_1layer_medium, rep = 'best', show.weights = FALSE)

plot(mod_1layer_medium, rep = 'best', show.weights = FALSE, intercept = FALSE)

### can try a different activation function!
set.seed(41231)
mod_1layer_medium_tanh <- neuralnet(compressive_strength ~ .,
                                    data = ready_df,
                                    hidden = 16,
                                    err.fct = 'sse',
                                    act.fct = 'tanh',
                                    stepmax = 5e5,
                                    linear.output = TRUE,
                                    likelihood = TRUE)

plot(mod_1layer_medium_tanh, rep = 'best', show.weights = FALSE)

### unfortunately, neuralnet isn't that great...for this problem we can't
### try a larger single hidden layer network...

### and even though neuralnet can handle 3 hidden layers...the fitting
### algorithm won't converge for larger networks in this example...

### but lets compare the neural networks we have fit so far 
### we included the likelihood formulation and so SIGMA was also estimated
### this means we can use AIC/BIC to compare the models!

tidy_neuralnet_results <- function(a_result, num_hidden_units)
{
  a_result %>% as.data.frame() %>% 
    tibble::rownames_to_column() %>% 
    tibble::as_tibble() %>% 
    slice(1:5) %>% 
    pivot_wider(names_from = 'rowname', values_from = 'V1') %>%
    # tidyr::spread(rowname, V1) %>%
    mutate(layer1 = num_hidden_units$layer1,
           layer2 = num_hidden_units$layer2,
           layer3 = num_hidden_units$layer3)
}

tidy_neuralnet_results(mod_1layer_small$result.matrix,
                       list(layer1 = 3, layer2 = 0, layer3 = 0))

purrr::map2_dfr(list(mod_1layer_small$result.matrix,
                     mod_1layer_small_b$result.matrix,
                     mod_1layer_medium$result.matrix,
                     mod_1layer_medium_tanh$result.matrix),
                list(list(layer1 = 3, layer2 = 0, layer3 = 0),
                     list(layer1 = 7, layer2 = 0, layer3 = 0),
                     list(layer1 = 16, layer2 = 0, layer3 = 0),
                     list(layer1 = 16, layer2 = 0, layer3 = 0)),
                tidy_neuralnet_results) %>% 
  arrange(aic, bic)

### let's use resampling to understand the overfitting behavior

### since neuralnet isn't that great, we will instead use nnet
### and have caret manage the resampling for us

library(caret)

### define the resampling scheme, let's just use 5 fold CV
### save the hold out set predictions in each fold
ctrl_cv05 <- trainControl(method = "cv", number = 5,
                          savePredictions = TRUE)

### nnet is simple it only allows 1 hidden layer but it can handle larger
### hidden layers than neuralnet!

### it also allows for REGULARIZATION!!!

### with neural networks it is critical to STANDARDIZE the inputs first!

### use a default tuning grid to start

set.seed(31311)
fit_nnet_default <- train(compressive_strength ~ ., 
                          data = my_concrete,
                          method = "nnet",
                          metric = "Rsquared",
                          preProcess = c("center", "scale"),
                          trControl = ctrl_cv05,
                          linout = TRUE,
                          trace = FALSE)

fit_nnet_default

### let's use a custom tuning grid! focusing on larger networks
### but the regularization via WEIGHT DECAY tries to limit overfitting
nnet_grid <- expand.grid(size = c(10, 20, 30, 40, 50),
                         decay = exp(seq(-6, 1, length.out = 11)))

nnet_grid %>% dim()

### this may take a LONG TIME to run!!!!

set.seed(31311)
fit_nnet_tune <- train(compressive_strength ~ ., 
                       data = my_concrete,
                       method = "nnet",
                       metric = "Rsquared",
                       tuneGrid = nnet_grid,
                       preProcess = c("center", "scale"),
                       trControl = ctrl_cv05,
                       maxit = 1000,
                       MaxNWts = 2500,
                       linout = TRUE,
                       trace=FALSE)

fit_nnet_tune

plot(fit_nnet_tune)

plot(fit_nnet_tune, xTrans = log)

fit_nnet_tune$bestTune

### look at the neural network

library(NeuralNetTools)

plotnet(fit_nnet_tune$finalModel)

### variable importances

### garson may give misleading results
garson(fit_nnet_tune$finalModel)

### olden is better than garson
olden(fit_nnet_tune$finalModel)

### unfortunately caret uses garson!!!
plot(varImp(fit_nnet_tune))

### study predictions!!!

### create a full factorial grid for the inputs based on the top 4 inputs
### ranked by the garson method

viz_grid <- expand.grid(age = unique(floor(seq(min(my_concrete$age),
                                               max(my_concrete$age),
                                               length.out = 25))),
                        cement = seq(min(my_concrete$cement),
                                     max(my_concrete$cement),
                                     length.out = 7),
                        blast_furnace_slag = unique(quantile(my_concrete$blast_furnace_slag,
                                                      seq(0, 0.8, by = 0.2))),
                        superplasticizer = unique(quantile(my_concrete$superplasticizer,
                                                    seq(0, 0.8, by = 0.2))),
                        water = median(my_concrete$water),
                        fine_aggregate = median(my_concrete$fine_aggregate),
                        coarse_aggregate = median(my_concrete$coarse_aggregate),
                        fly_ash = median(my_concrete$fly_ash),
                        KEEP.OUT.ATTRS = FALSE,
                        stringsAsFactors = FALSE) %>% 
  select(all_of(names(my_concrete %>% select(-compressive_strength))))

### why those bounds on blast_furnace_slag and superplasticer?
my_concrete %>% 
  ggplot(mapping = aes(x = superplasticizer, y = blast_furnace_slag)) + 
  geom_hline(yintercept = my_concrete %>% 
               pull(blast_furnace_slag) %>% 
               quantile(seq(0, 1, by = 0.2)),
             color = 'red',
             size = 1.2) +
  geom_vline(xintercept = my_concrete %>% 
               pull(superplasticizer) %>% 
               quantile(seq(0, 1, by = 0.2)),
             color = 'blue',
             size = 1.2) +
  geom_point(size = 2.5) +
  theme_bw()

viz_grid %>% dim()

viz_grid %>% glimpse()

viz_grid %>% head()

viz_grid %>% 
  mutate(pred = predict(fit_nnet_tune, newdata = viz_grid)) %>% 
  head()

viz_grid %>% 
  mutate(pred = predict(fit_nnet_tune, newdata = viz_grid)) %>% 
  ggplot(mapping = aes(x = age, y = pred)) +
  geom_line(mapping = aes(group = interaction(cement, 
                                              blast_furnace_slag,
                                              superplasticizer),
                          color = cement),
            size = 1.) +
  geom_hline(yintercept = my_concrete %>% 
               pull(compressive_strength) %>% 
               range(),
             color = 'red', linetype = 'dashed', size = 1.) +
  facet_grid(blast_furnace_slag ~ superplasticizer,
             labeller = 'label_both',
             as.table = FALSE) +
  scale_color_viridis_c(option = 'viridis') +
  theme_bw()

### visualize the predictions as a surface plot to examine how the
### predictions change wrt the top 2 inputs at the same time
viz_grid_2 <- expand.grid(age = seq(min(my_concrete$age),
                                    max(my_concrete$age),
                                    length.out = 51),
                          cement = seq(min(my_concrete$cement),
                                       max(my_concrete$cement),
                                       length.out = 51),
                          blast_furnace_slag = unique(quantile(my_concrete$blast_furnace_slag,
                                                        seq(0, 0.8, by = 0.2))),
                          superplasticizer = unique(quantile(my_concrete$superplasticizer,
                                                      seq(0, 0.8, by = 0.2))),
                          water = median(my_concrete$water),
                          fine_aggregate = median(my_concrete$fine_aggregate),
                          coarse_aggregate = median(my_concrete$coarse_aggregate),
                          fly_ash = median(my_concrete$fly_ash),
                          KEEP.OUT.ATTRS = FALSE,
                          stringsAsFactors = FALSE) %>% 
  select(all_of(names(my_concrete %>% select(-compressive_strength))))



viz_grid_2 %>% dim()

viz_grid_2 %>% count(cement, age) %>% 
  tibble::as_tibble()

viz_grid_2 %>% 
  mutate(pred = predict(fit_nnet_tune, newdata = viz_grid_2)) %>% 
  head()

### include horizontal reference lines to show the training set output
### lower and upper bounds
viz_grid_2 %>% 
  mutate(pred = predict(fit_nnet_tune, newdata = viz_grid_2)) %>% 
  ggplot(mapping = aes(x = age, y = cement)) +
  geom_tile(mapping = aes(fill = pred,
                          group = interaction(age, cement,
                                              blast_furnace_slag,
                                              superplasticizer))) +
  facet_grid(blast_furnace_slag ~ superplasticizer,
             labeller = 'label_both',
             as.table = FALSE) +
  scale_fill_viridis_c(option = 'turbo') +
  theme_bw()

### anything outside the range of the training set?
viz_grid_2 %>% 
  mutate(pred = predict(fit_nnet_tune, newdata = viz_grid_2)) %>% 
  ggplot(mapping = aes(x = age, y = cement)) +
  geom_tile(mapping = aes(fill = pred,
                          group = interaction(age, cement,
                                              blast_furnace_slag,
                                              superplasticizer))) +
  facet_grid(blast_furnace_slag ~ superplasticizer,
             labeller = 'label_both',
             as.table = FALSE) +
  scale_fill_viridis_c(option = 'turbo',
                       limits = range(my_concrete$compressive_strength)) +
  theme_bw()
