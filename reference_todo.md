TODO: 
- load in data
- normalize data
- build simple model
- run
- hyperparam tune/increase complexity of model if needed

---
title: "CS-137 Assignment 2: Convolutional Neural Networks for Weather Forecasting"
geometry: margin=2cm
output: pdf_document
---

## Goal

The goal of this assignment is to gain hands-on experience with **Convolutional Neural Networks (CNNs)** and to practice solving near-real-world problems using deep learning. You will work with real meteorological data to build a model that forecasts local weather conditions.

## Project Description

In this project, you will build a CNN-based model that takes a snapshot of the weather at time **t** and predicts weather conditions 24 hours later (**t + 24h**) at the grid point nearest to the location of the **Jumbo Statue at Tufts University** (42.40777867717294, -71.12041637590173).

### Input

The input **X_t** is a spatial snapshot of weather variables at time *t*, with shape:

```
(450, 449, c)
```

where `c` is the number of weather variables over the spatial area. The spatia average covers New England area of the US, which is shown by the figure below.    

![Spatial area of the study](fig1_region_map.png){width=300px}


### Prediction Targets

The model should predict the following six weather variables at the target grid point 24 hours ahead:

| Variable | Unit |
|---|---|
| `TMP@2m_above_ground` | K |
| `RH@2m_above_ground` | % |
| `UGRD@10m_above_ground` | m/s |
| `VGRD@10m_above_ground` | m/s |
| `GUST@surface` | m/s |
| `APCP_1hr_acc_fcst@surface` | mm |
| `APCP_1hr_acc_fcst@surface > 2mm` | binary label |

### Evaluation

- All continuous target variables are evaluated using **Root Mean Squared Error (RMSE)**.
- The RMSE for `APCP_1hr_acc_fcst@surface` is computed **only when the true value exceeds 2 mm**.
- The binary label (`APCP_1hr_acc_fcst@surface > 2mm`) is evaluated using the AUC (area under the ROC curve).

### Data Preparation

All the data are at `/cluster/tufts/c26sp1cs0137/data/assignment2_data`.  Currently the dataset covers **three years** of hourly weather data. We will add new data to the dataset. The data files are provided in the `dataset/` folder:

1. **Input tensors** — For each hour *t*, all weather variables over the spatial area are stored as a `torch.Tensor` of shape `(450, 449, c)` with dtype `bfloat16. Some input tensors contain NaN values. Please just neglect these tensors. The input tensors are in the `inputs` folder. Each file contains one hour of the data, and the file name indicates the hour.  
2. **Target tensors** — Target values at the Jumbo Statue grid point for all timesteps, including the binary precipitation label. The file is named target.pt
3. **Meta data** — Meta data of the dataset, including variable names, map projection, the grid point used as the target.  

To find more details of the data, please check the code (`assignment2_data/preparation/generate_dataset.py`) that generate the dataset.   

A Jupyter notebook is also provided with example visualizations, including:
- A map of the spatial region
- A plot of temperature over the spatial area
- A plot of average temperature over one year


## Grading (100 Points)

### Part 1 — Train a CNN (60 points)

Successfully train a CNN model that takes the spatial weather snapshot as input and predicts the six target variables 24 hours ahead. You are free to choose your architecture, but your model must:

- Accept inputs of shape `(450, 449, c)`
- Predict all six target variables for the Jumbo Statue location
- Be evaluated on the held-out test set using RMSE (and classification metric for the binary label)
- Include a brief write-up describing your architecture and training procedure

### Part 2 — Diagnosing Where the Weather Comes From (20 points)

Study and explain **which regions of the input map most influence the model's predictions**. Specifically:

- Compute and visualize the **gradient of the output with respect to the input** (saliency maps / sensitivity analysis)
- Discuss what the gradients reveal about which geographic areas or weather patterns drive the forecast
- Relate your findings to physical intuition about weather systems (e.g., prevailing wind directions, upstream regions)

### Part 3 — Independent Study at Choice (20 points)

Choose **one** of the following topics to investigate and report on:

| Option | Description |
|---|---|
| **Architecture choice** | Compare different CNN architectures (e.g., depth, kernel sizes, pooling strategies) and analyze their effect on forecast accuracy |
| **Convolutional kernel diagnosis** | Visualize and interpret learned convolutional filters; discuss what spatial patterns they detect |
| **Input variable study** | Perform ablation experiments to determine which input variables contribute most to predictive skill |
| **Residual connections (ResLinks)** | Study the effect of adding residual (skip) connections on model performance and training dynamics |
| **Pre-training from generic data** | Investigate whether pre-training the CNN on a broader dataset (e.g., a different region or time period) before fine-tuning improves results |

Your write-up for Part 3 should include clear experiments, results, and discussion.

## Working in Groups

Students **must** work in groups. The maximum group size is **3 students**. Each group submits a single report and codebase. All group members are expected to contribute and should briefly describe their individual contributions in the submission.

## Deliverables

- **Code**: A well-organized repository or zip file containing all training scripts and notebooks
- **Report**: A written report (PDF or Markdown) covering Parts 1, 2, and 3. 
- **Presentation**: A short presentation (up to 7 pages). 
- **Model checkpoint**: Your best trained model weights

## Due time 

The assignment is due at Apr. 3rd, 11:59pm. You will need to submit all deliverables to GradeScope. You will present your work at the class on Apr. 9th.    

*Good luck, and have fun forecasting the weather at Jumbo's location!*
