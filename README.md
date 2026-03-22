<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


<!-- PROJECT LOGO -->
<br />
<div align="center">

[//]: # (  <a href="https://github.com/Katsumi852/LiftedDynamics-UserStudy">)

[//]: # (    <img src="images/logo.png" alt="Logo" width="80" height="80">)

[//]: # (  </a>)

<h3 align="center">LiftedDynamics-UserStudy</h3>

  <p align="center">
    User Studies on the Dynamics of Lifted Objects During Haptic Interactions
    <br />
    <a href="https://github.com/Katsumi852/LiftedDynamics-UserStudy"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Katsumi852/LiftedDynamics-UserStudy">View Demo</a>
    &middot;
    <a href="https://github.com/Katsumi852/LiftedDynamics-UserStudy/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/Katsumi852/LiftedDynamics-UserStudy/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
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
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->

## About The Project

[//]: # ([![Product Name Screen Shot][product-screenshot]]&#40;https://example.com&#41;)

This study aims to evaluate the haptic rendering system for human interaction with lifted objects. During the
experiment, participants will first be introduced to the experimental apparatus and the virtual object with which they
will interact. They will then be asked to hold the stylus of the Touch X haptic device and shake it arbitrarily, as if
lifting and shaking the virtual object. The trajectory will be recorded by the haptic device, while our model computes
the corresponding force feedback in real time. The haptic device will then provide the force to the participant, who
will be asked to rate the haptic rendering feelings after shaking for a few seconds and observing how the virtual object
behaves during the experiment.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Python][Python.org]][Python-url]
* [![C++][C++.org]][C++-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->

## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running, follow these steps.

### Prerequisites

These are the tools and libraries you need to have installed before running the project:

* [OpenHaptics SDK](https://support.3dsystems.com/s/article/OpenHaptics-for-Windows-Developer-Edition-v35?language=en_US)
  for C++ program
* Python 3.11.x (optional)
    ```sh
    python --version
    ```
* Visual Studio 2019 (optional)

### Installation

#### Python Program Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/Katsumi852/LiftedDynamics-UserStudy.git
    cd LiftedDynamics-UserStudy
    ```
2. (Recommended) Create and activate a virtual environment:
    ```sh
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3. Install Python dependencies:
    ```sh
    pip install -r ./model_dynamics/requirements.txt
    ```

#### C++ Program Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/Katsumi852/LiftedDynamics-UserStudy.git
    cd LiftedDynamics-UserStudy
    ```
2. Install Visual Studio 2019 (or other IDEs) if not already installed.
3. Install
   the [OpenHaptics SDK](https://support.3dsystems.com/s/article/OpenHaptics-for-Windows-Developer-Edition-v35?language=en_US).
4. Open the C++ project in Visual Studio 2019 (or other IDEs).
5. Configure the project to reference the OpenHaptics library.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->

## Usage

1. **Start the C++ program [`./udp_client/udp_client/main.cpp`](./udp_client/udp_client/main.cpp)**
    - Initializes the haptic stylus position to a predefined center.
    - Creates a UDP client to send TouchX haptic device stylus data to the Python program.
2. **Start the Python program [`./model_dynamics/scripts/main.py`](./model_dynamics/scripts/main.py)**
    - Follow the on-screen instructions printed by the program.
    - Once you see `>> Running haptic rendering`, the program will actively listen for messages from the C++ UDP client.
3. When the user **clicks the button on the haptic stylus**, the running C++ program detects the button input and starts
   sending messages via UDP. The messages include warm-up messages at the beginning, followed by sending the positions
   of the user-held haptic stylus.
4. After receiving the messages, the running Python program computes the force feedback to be rendered and the object
   dynamics for the next time step, based on the received position representing the user's motion. The program then
   sends the force data via UDP and waits for a new UDP message for the next iteration.
5. Upon receiving the force data, the running C++ program renders the force back to the user via the haptic stylus and
   starts the next iteration of obtaining and sending position data.
6. When the user **clicks the button on the haptic stylus again** after interacting for some time, the running C++
   program will send a specific message denoting termination. The running Python program will then exit the real-time
   computation loop and proceed to **plot an animation of the virtual object's dynamics during the haptic interaction**.
   This provides the user with an intuitive understanding of what happened and how the object evolved.
7. **Close the C++ and Python programs** to completely terminate the processes.

[//]: # (_For more examples, please refer to the [Documentation]&#40;https://example.com&#41;_)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->

## Roadmap

- [ ] Investigate how humans perceive object softness
    - [ ] Vary bending stiffness or stretching stiffness
    - [ ] Use haptic illusions by changing mass, length, or other parameters
    - [ ] Identify regularities in shaking motions that humans typically use to better perceive softness
- [ ] Investigate how haptic rendering contributes to VR immersion
    - [ ] Compare only haptic rendering vs. only visual cues in a VR HMD vs. combined haptic and visual cues
    - [ ] Examine how the above different conditions affect haptic and visual illusions
    - [ ] Build VR scenes to effectively showcase combined haptic and visual demonstrations

See the [open issues](https://github.com/Katsumi852/LiftedDynamics-UserStudy/issues) for a full list of proposed
features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any
contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also
simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<a href="https://github.com/Katsumi852/LiftedDynamics-UserStudy/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Katsumi852/LiftedDynamics-UserStudy" alt="contrib.rocks image" />
</a>



<!-- LICENSE -->

## License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->

## Contact

Yutong Zhang - katsumi.zyt@gmail.com

Project
Link: [https://github.com/Katsumi852/LiftedDynamics-UserStudy](https://github.com/Katsumi852/LiftedDynamics-UserStudy)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

* This project is developed on top of [Lagrangian Graph Neural Network (LGNN)](https://github.com/M3RG-IITD/LGNN).
* Uses [OpenHaptics](https://www.3dsystems.com/haptics-devices/openhaptics) for TouchX haptic device control.
* Utilizes [JAX-MD](https://github.com/jax-md/jax-md) for molecular dynamics simulations.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/Katsumi852/LiftedDynamics-UserStudy.svg?style=for-the-badge

[contributors-url]: https://github.com/Katsumi852/LiftedDynamics-UserStudy/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/Katsumi852/LiftedDynamics-UserStudy.svg?style=for-the-badge

[forks-url]: https://github.com/Katsumi852/LiftedDynamics-UserStudy/network/members

[stars-shield]: https://img.shields.io/github/stars/Katsumi852/LiftedDynamics-UserStudy.svg?style=for-the-badge

[stars-url]: https://github.com/Katsumi852/LiftedDynamics-UserStudy/stargazers

[issues-shield]: https://img.shields.io/github/issues/Katsumi852/LiftedDynamics-UserStudy.svg?style=for-the-badge

[issues-url]: https://github.com/Katsumi852/LiftedDynamics-UserStudy/issues

[license-shield]: https://img.shields.io/github/license/Katsumi852/LiftedDynamics-UserStudy.svg?style=for-the-badge

[license-url]: https://github.com/Katsumi852/LiftedDynamics-UserStudy/blob/master/LICENSE

[//]: # ([product-screenshot]: images/screenshot.png)

<!-- Shields.io badges. You can a comprehensive list with many more badges at: https://github.com/inttter/md-badges -->

[Python.org]: https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54

[Python-url]: https://www.python.org/

[C++.org]: https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white

[C++-url]: https://isocpp.org/
