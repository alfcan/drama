<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<br />
<div align="center">
  <h3 align="center">DRAMA: Data-oRiented fAirness Mutation Analysis</h3>

  <p align="center">
    A framework designed to identify biases in datasets used for training machine learning models.
    <br />
    <a href="https://github.com/alfcan/drama"><strong>Explore the documentation »</strong></a>
    <br />
    <br />
    <a href="https://github.com/alfcan/drama/issues/new?labels=bug&template=bug-report---.md">Report a bug</a>
    ·
    <a href="https://github.com/alfcan/drama/issues/new?labels=enhancement&template=feature-request---.md">Request a Function</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

DRAMA is a framework designed to identify and mitigate biases in datasets used for training machine learning (ML) models. By using mutation operators, DRAMA introduces variations in datasets and evaluates the impact of these changes on fairness metrics, allowing the detection of unfairness symptoms before the model training phase.

Here are some key features of DRAMA:

- **Bias Identification in Datasets**: Application of mutation operators to introduce variations in data and measure how these affect fairness metrics.
- **Fairness Symptoms Evaluation**: Measurement of fairness metrics before and after mutations to effectively identify biases.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

This section lists the main technologies used to develop DRAMA:

- [![Python][python-shield]][python-url]
- [![Pandas][pandas-shield]][pandas-url]
- [![TensorFlow][tensorflow-shield]][tensorflow-url]
- [![Scikit-learn][scikit-learn-shield]][scikit-learn-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

Here is an example of how you can set up the project locally.
To get a local copy up and running, follow these simple steps.

### Prerequisites

There are no specific prerequisites for using DRAMA.

### Installation

1. Clone the Repository:
   ```sh
   git clone https://github.com/alfcan/drama.git
   ```
2. Install Dependencies:
   ```sh
   cd drama
   pip install -r requirements.txt
   ```
3. Run the Framework:
   ```sh
   cd src
   python main.py
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## Contributing

Contributions are what makes the open source community such an amazing place to learn, inspire and create. Any contributions you make will be **very much appreciated**.

If you have a suggestion that could improve this project, please fork the repository and create a pull request. You can also simply open an issue with the ‘enhancement’ tag.
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m ‘Add some AmazingFeature’`)
4. Push on the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Alfonso Cannavale - [LinkedIn](https://www.linkedin.com/in/alfonso-cannavale-62150b229/) - [alfonsocannavale.it](http://alfonsocannavale.it)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/alfcan/drama.svg?style=for-the-badge
[contributors-url]: https://github.com/alfcan/drama/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/alfcan/drama.svg?style=for-the-badge
[forks-url]: https://github.com/alfcan/drama/network/members
[stars-shield]: https://img.shields.io/github/stars/alfcan/drama.svg?style=for-the-badge
[stars-url]: https://github.com/alfcan/drama/stargazers
[issues-shield]: https://img.shields.io/github/issues/alfcan/drama.svg?style=for-the-badge
[issues-url]: https://github.com/alfcan/drama/issues
[license-shield]: https://img.shields.io/github/license/alfcan/drama.svg?style=for-the-badge
[license-url]: https://github.com/alfcan/drama/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/alfonso-cannavale-62150b229/
[python-shield]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[python-url]: https://www.python.org/
[pandas-shield]: https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white
[pandas-url]: https://pandas.pydata.org/
[tensorflow-shield]: https://img.shields.io/badge/tensorflow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white
[tensorflow-url]: https://www.tensorflow.org/
[scikit-learn-shield]: https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white
[scikit-learn-url]: https://scikit-learn.org/
