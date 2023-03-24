<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">


  <h1 align="center">Convolutional Neural Networks for Visual Recognition - Assignment Solutions </h1>


</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary><h3>Table of Contents</h3></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li><a href="#assignment-1">Assignment 1</a></li>
    <li><a href="#assignment-2">Assignment 2</a></li>
    <li><a href="#assignment-3">Assignment 3</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project



This repository contains my solutions to the assignments for course : **AI fundamentals and Deep Learning Frameworks** offered by University of Maryland [Fall 2022]. This course is
based upon the famous Stanford Course [CS231n](http://cs231n.stanford.edu/) "Convolutional Neural Networks for Visual Recognition".

Stanford's CS231n is one of the best ways to dive into Deep Learning in general, in particular, into Computer Vision. Inline questions are explained in detail, the code is brief and commented.
And most of the solutions are optimized by using ```vectorization``` in numpy.
Assignments are completed using ```Pytorch```.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Assignment 1 -->
## Assignment 1

### Image Classification | kNN | SVM | Softmax | Neural Network

Q1: [k-Nearest Neighbor Classifier](https://github.com/KACHAPPILLY2021/stanford-cs231n-assignments/blob/main/assignment1/knn.ipynb)
- Test accuracy on CIFAR-10 : 28.2%

Q2: [Training a Support Vector Machine](https://github.com/KACHAPPILLY2021/stanford-cs231n-assignments/blob/main/assignment1/svm.ipynb)
- Test accuracy on CIFAR-10 : 37.1%

Q3: [Implement a Softmax classifier](https://github.com/KACHAPPILLY2021/stanford-cs231n-assignments/blob/main/assignment1/softmax.ipynb)
- Test accuracy on CIFAR-10 : 35.1%

Q4: [Two-Layer Neural Network](https://github.com/KACHAPPILLY2021/stanford-cs231n-assignments/blob/main/assignment1/two_layer_net.ipynb)
- Test accuracy on CIFAR-10 : 50.3%

Q5: [Higher Level Representations: Image Features](https://github.com/KACHAPPILLY2021/stanford-cs231n-assignments/blob/main/assignment1/features.ipynb)
- Test accuracy on CIFAR-10 : 59.5%

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Assignment 2 -->
## Assignment 2

### Fully-Connected Nets | Batch Normalization | Dropout | Convolutional Nets

Q1: [Fully-connected Neural Network](https://github.com/KACHAPPILLY2021/stanford-cs231n-assignments/blob/main/assignment2/FullyConnectedNets.ipynb)
- Validation / test accuracy on CIFAR-10 : 51.3% / 48.5%

Q2: [Batch Normalization](https://github.com/KACHAPPILLY2021/stanford-cs231n-assignments/blob/main/assignment2/BatchNormalization.ipynb)

Q3: [Dropout](https://github.com/KACHAPPILLY2021/stanford-cs231n-assignments/blob/main/assignment2/Dropout.ipynb)

Q4: [Convolutional Networks](https://github.com/KACHAPPILLY2021/stanford-cs231n-assignments/blob/main/assignment2/ConvolutionalNetworks.ipynb)

Q5: [PyTorch](https://github.com/KACHAPPILLY2021/stanford-cs231n-assignments/blob/main/assignment2/PyTorch.ipynb)  on CIFAR-10
- PyTorch implementation, test accuracy for designed architecture : 72.58%
    - ( **CONV** &rarr; **BatchNorm**  &rarr; **LeakyReLu** &rarr; **Maxpool** ) x2 &rarr; *Dropout* &rarr; *FC*


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Assignment 3 -->
## Assignment 3

### Image Captioning with Vanilla RNNs

Q1: [Image Captioning with Vanilla RNNs](https://github.com/KACHAPPILLY2021/stanford-cs231n-assignments/blob/main/assignment3/RNN_Captioning.ipynb)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started
- The official [course website](http://cs231n.stanford.edu/) 
- Video-lectures. Prerequisites are given in the 1st lecture.
	- Winter 2016 [YouTube playlist](https://www.youtube.com/playlist?list=PLLvH2FwAQhnpj1WEB-jHmPuUeQ8mX-XXG)
	- Spring 2017 [YouTube playlist](https://goo.gl/pcj7c8)

### Prerequisites
For some parts of the 3rd assignment need GPUs :
* [Kaggle Kernels](https://www.kaggle.com/code) or [Google Colaboratory](https://colab.research.google.com/) will do.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Jeffin Johny K - [![MAIL](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:jeffinjk@umd.edu)
	
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://kachappilly2021.github.io/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](http://www.linkedin.com/in/jeffin-johny-kachappilly-0a8597136)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See [MIT](https://choosealicense.com/licenses/mit/) for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com
