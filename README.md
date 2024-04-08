# Dynamic Submodular-Based Learning Strategy in Imbalanced Drifting Streams for Real-Time Safety Assessment in Nonstationary Environments

This repository contains the implementation of a dynamic submodular-based learning strategy with activation interval (DSA-AI) for real-time safety assessment in imbalanced drifting streams. The proposed DSA strategy aims to address the challenges posed by nonstationary environments by incorporating submodular optimization techniques into the learning process.

## Overview

Safety assessment in dynamic and imbalanced data streams is crucial for various applications, including autonomous systems and industrial monitoring. However, traditional approaches often struggle to adapt to changing environments and imbalanced class distributions, leading to degraded performance and reliability.

Our approach introduces a dynamic submodular-based learning strategy tailored for real-time safety assessment in nonstationary environments. By leveraging submodular optimization, our strategy effectively selects informative data points while considering the imbalanced nature of the data and adapting to concept drifts over time.


<img width="570" alt="截屏2024-04-08 20 44 09" src="https://github.com/liuzy0708/DSLS-Demo/assets/115722686/0b8eda46-f5f4-4851-952b-95544818b9b9">


<img width="588" alt="截屏2024-04-08 20 44 18" src="https://github.com/liuzy0708/DSLS-Demo/assets/115722686/0dd4411b-7531-4f64-88a9-fff82606d804">

## Key Features

- Dynamic submodular-based learning strategy with activation interval.
- Handling imbalanced class distributions.
- Adaptation to concept drifts in real-time.
- Integration with existing streaming algorithms.
- Efficient and scalable implementation.

## Installation

To utilize the DSA-AI strategy, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/your_username/your_repository.git
cd your_repository
```

2. Install dependencies:

```bash
conda env create -f environmental.yml
conda activate <environment_name>
```

3. Refer to the usage instructions in the documentation to integrate the DSA strategy into your workflow.

## Usage

This repository includes two main files:

1. **Demo.py**: This file demonstrates the application of the dynamic submodular-based learning strategy for real-time safety assessment in imbalanced drifting streams. It showcases the usage of the `Def_DSA_AI` implementation code. After configuring the environment, execute `Demo.py` to observe the DSA strategy in action.

2. **Def_DSA_AI.py**: This file contains the implementation of the DSA-AI strategy. The DSA-AI strategy selects and labels informative data points in imbalanced drifting streams using submodular optimization techniques.

To run the demo:

1. Clone the repository:

```bash
git clone https://github.com/your_username/your_repository.git
cd your_repository
```

2. Set up the conda environment using the provided environmental.yml file:

```bash
conda env create -f environmental.yml
conda activate <environment_name>
```

3. Ensure all dependencies are installed and the environment is configured properly.

4. Execute the `Demo.py` file:

```bash
python Demo.py
```

5. The demo will showcase the dynamic submodular-based learning strategy for real-time safety assessment. Customize the demo according to specific requirements or tasks.

For detailed explanations and customization options, refer to the comments and documentation within the source code files.

## Contributing

Contributions are appreciated! If you have suggestions, bug fixes, or new features, please open an issue or submit a pull request following our [contribution guidelines](CONTRIBUTING.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you find this repository or the implemented scheme useful in your research or work, please consider citing:

```
@ARTICLE{10195233,
  author={Liu, Zeyi and He, Xiao},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Dynamic Submodular-Based Learning Strategy in Imbalanced Drifting Streams for Real-Time Safety Assessment in Nonstationary Environments}, 
  year={2024},
  volume={35},
  number={3},
  pages={3038-3051},
  keywords={Safety;Annotations;Task analysis;Data models;Learning systems;Uncertainty;Adaptation models;Broad learning system (BLS);concept drift;nonstationary environments;real-time safety assessment (RTSA);submodular},
  doi={10.1109/TNNLS.2023.3294788}}
```

## Contact

For inquiries about the DSA-AI strategy, contact [liuzy21@mails.tsinghua.edu.cn](mailto:liuzy21@mails.tsinghua.edu.cn).

## Acknowledgments

We extend our sincere gratitude to the THUFDD Group, led by Prof. Xiao He and Prof. Donghua Zhou, for their invaluable support and contributions to the development of this scheme.

---

**Disclaimer:** Use this strategy at your own risk. No warranty is provided.
