# TL
# Lobby Game Recommendation System 2.0 (LRS 2.0)

# Description
- This project was created at 2023-11-14
- Key words: Lobby Machine Recommendation, Collaborative Filtering, MasedMultiHeadAttention, Transformer, ConsistentHashRing
- This project belongs to Hard Rock Games

# Outline
- [Objective](#objective)
- [Synthesis](#synthesis)
- [Data Extraction](#data-extraction)
- [Feature Map](#feature-map)
- [Feature Engineering](#feature-engineering)
- [Model Selection](#model-selection)
- [Development Processes](#Development-Processes)
- [Results and Performance](#results-and-performance)
- [Interpretability and Insights](#interpretability-and-insights)
- [Business Impact](#business-impact)
- [Installation](#installation)
- [Usage](#usage)
- [Authors](#authors)




# Objective

Generally speaking, the fundamental objective of this project is exploring a method to make recommendations from available slots that the player is most likely to enjoy. For specific details, please refer to lobby game recommendation system 1.0.

This project is an enhancement of 1.0:
- upgrading the deep leanring algorithms from Deep Auto Encoder to Transformer;
- introducing content-based features from game design level;
- introducing uuid as an importnat feature to make the model remember the preference of specifc users;
- introducing seasonal features from temporal dimension;
- introducing the concept of "Path" to make models understand behaviour sequence patterns;


  
| Comparison                    | LRS 1.0     | LRS 2.0 |
|:-------------------------|:----------|:---------|
| Algorithm      | DeepAutoEncoder+CollaborativeFiltering      | TransformerBased BehaviourSequence         |
| Feature Status           | Static      | Static + Dynmaic        |
| Unit of Data Aggregation | Week      | Path        |
| Temporal Features          | No      | Yes        |
| Game Content-based Features          | No      | Yes        |
| Play Behaviour  Features          | Yes     | Yes        |
| Machine Bias Score Function          | Not Normalized   | Normalized        |
| Model Scale | Small     | Large        |
| Recommendations for new uid | Default Score     | Personalized        |
| Support Real-time Pipeline | No     | Yes        |


The following list contains the requirements in more detail:
- 2023-11-28: Dmitry -> our data pipelines don't support cache-queue to record the most recent play actions currently, so transferring data as seq-2-seq format should be involved in the functions
- 2024-01-16: Matthew -> machine bias coefficient formula needs to be refined: " a new player spinning more than say around 50 times period shows a strong positive bias 10++
but a veteran clan player with an ltv of $5k spinning only 100 times in a specific game or on a specific day might be a very negative sign of experience and enjoyment"  



# Synthesis 

First and foremost, the system goes through the following process to make recommendations for one user:

- Retrain to Update Model on the first day of each month
- Fetch Last Day Data of the Target User as Input
- Load Input and Generate Machine Bias Coefficients by Day as Cache
- Pick Up Machines from Cache by Stratified Weighted Random Pickup (refer to LRS 1.0)

## -- When and How to retrain the models

<div align="center">
   <img src="/a.png" alt="Example Image">
</div>

The figure above represents a time-partitioned database from which training samples are extracted to update the system. 

The update process is scheduled to occur on the first day of each month, utilizing data from three distinct user groups:

$$U_{TargetMonth} = U_{LastMonth}  \cup  U_{TargetMonthOneYearAgo}  \cup  U_{TargetMonthTwoYearAgo}$$

- Recent Active Users: Users who had more than 2 active days in the previous month. This group provides the model with recent behavior patterns and trends.
- Yearly Seasonal Users (Last Year): Users who had more than 2 active days in the corresponding month of the previous year. This group helps the model understand and adjust for seasonal variations and annual trends by comparing the activity in the same period one year prior.
- Yearly Seasonal Users (Year Before Last): Users who had more than 2 active days in the corresponding month two years prior. Including this group allows the model to validate consistency or detect changes in long-term trends or recurring patterns.
By incorporating data from these diverse time frames, the model is likely aiming to achieve a balanced understanding of both short-term behaviors and long-term trends.

This strategy could be particularly beneficial in exploring the most recent behaviour patterns and also in scenarios where user behavior is influenced by factors such as seasonality, market trends, or product changes.

## -- When and How to Update Cache

The system generates two primary categories of cache by day:

<div align="center">
   <img src="/b.png" alt="Example Image">
</div>


- Personal Cache: This cache is tailored for users who have had at least one active day in the past 30 days. It is derived from the model output mentioned previously, incorporating data specifically related to individual user behaviors and activities.
- Default Cache: This cache is designed for users who have had no active days in the past 30 days. It is created by aggregating machine bias coefficients from all users who were active at any point during the month. This aggregation aims to provide a generalized set of data that can still offer relevant insights for users without recent activity.

Once generated, the results from both cache types are meticulously organized. They are sorted, grouped, and then stored in Redis as JSON (refer to LRS 1.0) shown in above figure.
This structured storage allows for efficient retrieval and utilization of the cache data, ensuring that the system can quickly access and serve the personalized or default information as needed.

# Data Extraction

## The Concepts of Step and Path

<div align="center">
   <img src="/e.png" alt="Example Image">
</div>

Initially, we introduced the concepts of Step and Path as mechanisms to capture and organize data. 

A Step is defined as a tuple consisting of machine ID, Date, and machine Type, arranged chronologically by timestamp. 

Conversely, a Path represents a series of these Steps, sequenced in the order of their timestamps to maintain the continuity of events.

<div align="center">
   <img src="/c.png" alt="Example Image">
</div>

The extraction process of raw data is shown in the figure above. For more details, please check sql files


# Feature Map

There are 3 clasess for all training features {temporal, content, user}
  

- temporal: Used to represent date features

- content: Used to represent machine s' content-based features including arts or game design level

- user: Used to represent profile or gameplay experience


## basic feature map
  
| Name                    | Type     | Class | Example              | Description                                                |
|:-------------------------|:----------|---------|----------------------------------|:------------------------------------------------------------|
| uuid            | string      | user         | NC-tD3MaqXQsJ | User ID associated with the user.                |
| mid             | string      | user        | pp              | machine ID associated with machine |
| dat             | string      | temporal        |  1/1/2022 10:10 UTC             | login date |
| path          | int      | user        | 10              | order of step group (playing under same game, typ, dat)           |
| slv             | int      | user        | 10                | spending lv        |
| lt              | int      | user        | 10                   | lifetime by day                   |
| typ             | int      | user        | 1          | lobby/clan/jounery             |
| mod             | int      | user        | 1          | jackpot/simple/team/sit&go/social             |
| uas          | bool      | user        | 1             | if using autospin within this path                |
| abbr          | float      | user        | 1.0              | avg bet/balance within this path           |
| ubbr         | float      | user        | 1.0           | max bet/balance within this path                       |
| lbbr     | float      | user         | 1.0          | min bet/balance within this path          |
| hbc     | bool         | user        | 1                   | if using bought coins within this path                         |
| times     | int         | user        | 1                   | count spins within this path                         |


## add-on feature map
  
| Name                    | Type     | Class | Example              | Description                                                |
|:-------------------------|:----------|---------|----------------------------------|:------------------------------------------------------------|
| hoilday            | string      | temporal         | new year | hoilday of dat               |
| weekday             | int      | temporal        | 1             | weekday of dat |
| theme1             | string      | content        |  Fantasy             | machine theme defined by artist   |
| theme2          | string      | content        | Cartoon              | machine theme defined by artist            |




# Feature Engineering

## Observation Duration
| Type | RateLeft1DReturn | RateLeft2DRetur | RateLeft3DReturn | RateLeft7DReturn | RateLeft14DReturn |RateLeft21DReturn|
|-----------------|----------|---------|------------------|-------------|-------------|-------------|
| first month users| 0.76 | 0.65 | 0.57 | 0.37 | 0.18 |0.08|

## Machine Bias Score
The following function is using in LRS 1.0, where TS is the total spin times, NMP is unique machine, and MS is the each machine's spin times

$$ BC = \sqrt[3]{\log_2{TS} \cdot (\log_2{NM} + 1) \cdot \left( \frac{MS}{TS} \right)} $$

A significant limitation of this approach is the absence of predefined upper and lower bounds for each parameter. 

Additionally, it does not take into account the varying weight of each variable, leading to a final outcome whose maximum value cannot be predicted or constrained.

To address these issues, we have developed a specific formulaic approach. 
Initially, we determine the maximum threshold for each parameter by discarding outliers to ensure that no value exceeds this threshold. 

Subsequently, we implement min-max normalization to ensure all parameters are confined within the [0,1] interval. 

Lastly, we assign a tailored weight to each parameter, with the collective weight of all parameters equating to 1. 
This allows us to amplify the weighted sum so that it falls within the [LB, UB] range.

$$ \begin{Bmatrix}
x' = max(x, \quad MaxNoOutliers(x)) \\
F_{norm} = \frac{x' - \min(x')}{\max(x') - \min(x')} \\
Score = LB + (UB-LB) * \sum_{p} weight_{p} * F_{norm}(p) \quad|\quad \forall p \in U_{params} \\
\end{Bmatrix} $$

The parameters used in this project are {TotalSpins, PathSpins, UniqueMachine, RepeatMachineTimes} with weights of {0.1, 0.65, 0.1, 0.15}

## Encoder

Three different coding techniques were used in this project:

For string type vals:
One thing to explain is that the number of month active users may reach 1e7, so we need to use partition technique on uuid
- Unique Value Map {key: count}: {mid, theme1, theme2, holiday}
- Consistent Hash Ring to get chunk and Put {key: global count} in corresponding chunk: {uuid}

For all int type vals: 
- int(log2(x)), int(log10(x))

## Seq-2-Seq Transformation

<div align="center">
   <img src="/d.png" alt="Example Image">
</div>

The figure indicates the required data handling process for our model: 
- data for a single user spanning the timeframe of t-1 days is organized into an array by the sequence of actions taken. This sorted array serves as the input
- Concurrently, the bias score corresponding to the user's activities on day t is computed to function as the output.

Is worthy to mention that we fix the max length of input sequence as 50, so we only extract the last 50 pathes for those extremely active users

# Model Selection

This project uses a Transformer-based network architecture that mixes dynamic (sequence) and static data features.
At a high level, the network works in the following steps：
- Embedding Static Features and Concat Them (Batch, num_static_features) -> (Batch, embedding_size * num_static_features)
- Embedding Dynamic Features and Stack Them (Batch, Sequence_length, num_dynmaic_features) - > (batch, seq, feature_num, embed_size) -> (batch, seq, embed_size * feature_num)
- Foward Stack Embedded Dynamic Features into the part of MultiLayer(Masked MultiHeadAttestionLayers + PositionWiseFeedForwardNetworks)
- Avg Pooling dynamic tensor + static tensor
- Foward into PositionWiseFeedForwardNetworks
- Foward into number of available lobby games
  
## Network Architecture
<div align="center">
   <img src="/g.png" alt="Example Image">
</div>

## Challenge
- split static and dynmaic tensor in batch and pay attention to the order of both
- only adding positional encoding to dynmaic tensor
- get mask_pad_value for multi-dynmaic tensor
- how to connect static and dynmaic tensor at fully connection layer, since they have different shape
- using layer normalization but not batch normalzaiton
- Data Augmentation for uuid to let the model predict new user
- design better loss function


# Development Processes

<div align="center">
   <img src="/h.png" alt="Example Image">
</div>

## env
- python3.10
- MPS(GPU)
## hyperparameters
- batch: 32
- lr: 1e-3
- d_model: 128
- num_layer: 3
- max_seq_lengh: 50 (max_num of paths covered)


# Results and Performance

## How to choose Loss Function
This project is framed as a multi-label regression problem, which involves predicting multiple continuous target variables. 

For such problems, common loss functions used to measure the discrepancy between the predicted values and the actual values include:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

The objectives of our project do not align with traditional multi-label regression techniques. 
As our aim is centered around generating tailored recommendations for users, specifically targeting machines they might prefer the next day. This involves two key components: 
- firstly, forecasting the user's bias score for the game they were actual played at the target day
- secondly, identifying and suggesting alternative games that the user didn't played at the target but could potentially find appealing.

## Masked Mean Absolute Error

The introduction of Masked Mean Absolute Error (MMAE) in this scenario is a tailored solution addressing the unique requirements of the project. 
This approach ingeniously incorporates a masking technique to focus the error calculation specifically on relevant predictions. Here's how it works:

- Masking: A mask is applied to the dataset to selectively ignore the machines that a player has not interacted with. This ensures that the error calculation is only concentrated on the machines that the player has actually played.
- MAE Calculation: Once the mask is in place, the Mean Absolute Error (MAE) is computed, but it's restricted to the predictions corresponding to the games that the player has actually played. This way, the error metric truly reflects the accuracy of predictions for the user's real interactions, rather than being skewed by irrelevant data.
  
By focusing solely on the games that the player engages with, the Masked Mean Absolute Error provides a more accurate and meaningful measure of the predictive performance of the recommendation system, making it a highly suitable approach for this specific context.

<div align="center">
   <img src="/f.png" alt="Example Image">
</div>

## Log Performance
We listed the part of training log (for more details pls check ./log)

After only 5 training runs, the MMAE loss on the test set has converged to around 0.14.

Given that the actual bias coefficient falls within the range of 0.1 to 1, it implies that for a true value of 0.5, our predicted values would typically lie between 0.36 and 0.64.

--------------------------------------------------------------------
2024-02-04 18:30:24 PM - log_lgrs - INFO -main.py - Running Under mps

2024-02-04 18:30:37 PM - log_lgrs - INFO -main.py - Batch:1 | Loss:1.0788023471832275

2024-02-04 18:47:02 PM - log_lgrs - INFO -main.py - Epoch 0: New best model saved with loss 0.16316839207969389

2024-02-04 19:03:36 PM - log_lgrs - INFO -main.py - Epoch 1: New best model saved with loss 0.14936022788317696

2024-02-04 19:20:32 PM - log_lgrs - INFO -main.py - Epoch 2: New best model saved with loss 0.14784992833774807

2024-02-04 19:37:45 PM - log_lgrs - INFO -main.py - Epoch 3, Test Loss: 0.14821464416452393

2024-02-04 19:55:00 PM - log_lgrs - INFO -main.py - Epoch 4, Test Loss: 0.146596919601548

2024-02-04 19:55:00 PM - log_lgrs - INFO -main.py - Epoch 4: New best model saved with loss 0.146596919601548

--------------------------------------------------------------------

# Interpretability and Insights




# Installation

using docker to build imdb and run


# Usage
Set targetDate in main.py

# Authors
- Wayne Wu: wayne.wu@hardrockdigital.com

