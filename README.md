# Project

Welcome to our Data and Model Drift Repository! Things in our world are permanently changing. For machine learning, this means that productive models are confronted with unknown data and can become outdated. A proactive drift management approach is required to ensure that productive AI services deliver consistent business value over the long term.

Please check out our background article [Getting traction on Data and Model Drift with Azure Machine Learning](https://medium.com/p/ebd240176b8b/edit) for an in-depth discussion about the concepts used in this repo.

Starting with tabular data use cases, we provide the [following samples](tabular-data/DATA_MODEL_DRIFT.ipynb) to detect and mitigate data and model drift:

### 1. Statistical tests and expressive visualizations to detect and analyze drift in features and model predictions

<img src="media/data-drift-kde-short.png" alt="KDE intersections to identify data drift" width="600"/>


### 2. A predictive approach to identify the impact of data and concept drift on the model

<img src="media/probas-combined.png" alt="Model drift impact on predicted class probabilities" width="600"/>

### 3. Sample code for creating automated Pipelines to identify data drift regularly as part of an MLOps solution using Azure Machine Learning

<img src="media/evergreen-mlops.png" alt="MLOps architecture for evergreen models" width="600"/>

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
