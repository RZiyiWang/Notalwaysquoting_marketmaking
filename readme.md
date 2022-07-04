# Robust Market Making: To Quote, or not To Quote


## User Guide

   ### Instructions to configure the environment:

1. Install the basic python3 environment and VS code.

2. Intsall the Jupyter Notebook extension in VS code.

3. Install necessary libraries as follows:

   | Library    | Version |
   | ---------- | ------- |
   | numpy      | 1.21.5  |
   | TensorFlow | 2.8.0   |
   | tf_agents  | 0.9.0   |
   | matplotlib | 3.5.1   |
   | seaborn    | 0.11.2  |

   ### A brief description of the modules

1. adversaryEnv: four classes of adversarial environment
2. marketMakerEnv: four classes of always quoting market makers in accordingly four adversarial environments, plus market makers in fixed and random environments
3. dynamics: Execution dynamics based on Poisson distribution, price dynamics based on Brownian Motion with drift
4. strategies: An market making strategy that calculate bid and ask offset based on current inventory and prices at different levels of risk aversion
5. agents: Soft actor-critic reinforcement learning agent （suitable for always quoting MMs and adversaries）
6. constants：Hyperparameters （suitable for always quoting MMs and adversaries）
7. utils：Evaluating the performance of the SAC-agents
8. allowNotQuoteEnv: allowing not to quote’s market-making environment 
9. randomenvtest: To validate our environment we will use a random policy to generate actions.
————————————————————————————————————
10. 7 files starting with 'RN_' and 'RA_’: ‘RN.ipynb’ file refers to market makers who are adversarially trained to quote forever in a risk neutral situation and RA refers to market makers who are trained to quote forever based on 6 different sets of risk averse parameters.
11. trainNotQuoteMM: Training allowing not to quote MM agent using DQN algorithm (This is a template, and you need to change the parameters according to different environments, such as the name of the adversarial environment, or the risk aversion parameter. Alternatively, you can just use files named starting with ‘Quote_’ , such as ‘Quote_a.ipynb’.)

   ### Instructions to run the codes

For example, if we want to train a market maker against a strategic adversary with controlling drift in the RN case, we need to complete the following two steps: 

1. Open "RN.ipynb", find the block "RN_train Agent:MM VS StrategicAdversaryWithControllingDrift" .First train the adversary, then train the corresponding market maker. 

2. Open "Quote_b.ipynb" and train a market maker that allows no to quote.