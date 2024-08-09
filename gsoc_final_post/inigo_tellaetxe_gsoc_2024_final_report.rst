.. role:: raw-html(raw)
   :format: html

.. raw:: html

   <center><a href="https://summerofcode.withgoogle.com/programs/2024/projects/BpY78ovV"><img src="https://developers.google.com/open-source/gsoc/resources/downloads/GSoC-logo-horizontal.svg" alt="gsoc" height="50"/></a></center>

.. raw:: html

   <center>
   <a href="https://summerofcode.withgoogle.com/programs/2024/projects/BpY78ovV"><img src="https://www.python.org/static/community_logos/python-logo.png" height="50"/></a>
   <a href="http://dipy.org"><img src="https://python-gsoc.org/logos/DIPY.png" alt="dipy" height="50"/></a>
   </center>


Google Summer of Code Final Work Product
========================================

-  **Name:** Iñigo Tellaetxe Elorriaga
-  **Organization:** Python Software Foundation
-  **Sub-Organization:** DIPY
-  **Project:** `DIPY-Conditional Tractogram Generation Based on age and neurodegeneration using autoencoder networks <https://github.com/dipy/dipy/wiki/Google-Summer-of-Code-2024#project-5-project-ideas-using-aiml-in-diffusion-mri-processing>`_
-  **Repository:** `TractoEncoder GSoC <https://github.com/itellaetxe/tractoencoder_gsoc>`_


Proposed Objectives
-------------------

* Replicate the results of the AutoEncoder architecture in the `GESTA <https://doi.org/10.1016/j.media.2023.102761>`_ paper using TensorFlow2 as the deep learning backend.
* Obtain publicly available data from the `ADNI <http://adni.loni.usc.edu/>`_, the `FiberCup <https://tractometer.org/fibercup/home/>`_ and the `ISMRM2015 tractography challenge <https://tractometer.org/ismrm2015/home/>`_ to compute tractograms for training the deep learning models.
* Integrate the replicated unconditional AutoEncoder architecture and sampling algorithm into DIPY to generate synthetic human tractograms.
* Investigate how to condition the AutoEncoder architecture on scalar and categorical variables to support conditional tractogram generation on age and neurodegeneration status.
* Implement a conditional AutoEncoder architecture that can generate synthetic tractograms conditioned on age and neurodegeneration status.


Modified Objectives(Additional)
-------------------------------

* Investigate the use of Variational AutoEncoders (VAE) for unconditional tractogram generation based on the GESTA architecture.
* Investigate the use of Conditional Variational AutoEncoders (CVAE) for conditional tractogram generation based on the GESTA architecture.
* Investigate the possibility to condition the tractogram generation on the fiber bundle for additional control over the process and the data generation.

Objectives Completed
--------------------

* Conducted Literature Review on synthetic tractography generation using AutoEncoders. Main inspirations: 

  * The `FINTA <https://10.1016/j.media.2021.102126>`_ and `GESTA <https://doi.org/10.1016/j.media.2023.102761>`_ papers because they provide a relatively simple AE architecture with an open source sampling algorithm, easy to reuse.

  * `Variational AutoEncoders for Regression <https://doi.org/10.1007/978-3-030-32245-8_91>`_, which provided a good starting point for conditional Variational AutoEncoders with direct application to brain aging, related to the project's objectives.

  * `Attribute-based Regularization of latent spaces for variational AutoEncoders <https://doi.org/10.1007/s00521-020-05270-2>`_ and `Adversarial AutoEncoders <http://arxiv.org/abs/1511.05644>`_, which inspired the use of Attribute-based regularization for conditioning on a continuous variable (age); and the use of adversarial training for conditioning on categoric variables (fiber bundle or neurodegeneration status), respectively.

* Replicated the `GESTA <https://doi.org/10.1016/j.media.2023.102761>`_ architecture originally implemented in PyTorch (vanilla AutoEncoder, not variational) using TensorFlow2+Keras. Validated the results using the FiberCup dataset. The model is found in the ``ae_model.py`` module of the `TractoEncoder GSoC <https://github.com/itellaetxe/tractoencoder_gsoc>`_ repository.
  * Weight and bias initializers are different in PyTorch compared to TensorFlow2, so the PyTorch behavior was replicated using custom initializers. Different weight & bias initialization strategies can lead to drastically different training results, so this step took extra care.
  * The upsampling layers of the Decoder block use linear interpolation in PyTorch by default, whereas in TensorFlow2, there is no native implementation for this, and nearest neighbor (NN) interpolation is used instead. This is a significant difference in implementations, and to replicate the PyTorch behavior a custom linear interpolating upsampling layer was implemented in TF2. However, after training with both NN and linear interpolation, the results were very similar, so the custom layer was not used in the final implementation.
  * The figure below shows a summary of the replication results:

  .. image:: https://github.com/dipy/dipy.org/blob/master/_static/images/gsoc/2024/inigo/fibercup_replicated.png
    :alt: Replication of the GESTA architecture results on the FiberCup dataset.
    :width: 800

* Implemented a Variational AutoEncoder (VAE) architecture based on the `GESTA <https://doi.org/10.1016/j.media.2023.102761>`_ architecture. The model is found in the ``vae_model.py`` module of the `TractoEncoder GSoC <https://github.com/itellaetxe/tractoencoder_gsoc>`_ repository.
  * It was necessary to implement Batch Normalization after the Encoder convolutional layers and exponential operator clipping to prevent gradient explosion while training the model. This modification allowed stable training and it was a major contribution to the robustness of the architecture.
  * Weighing the Kullback-Leibler loss component was also implement, based on the`Beta-VAE <>`_ work, aiming for a stronger disentanglement of the latent space. However, this parameter was never explored (it was always set to 1.0) due to its trade-off with the reconstruction accuracy, but it is a potential improvement for future work in the context of hyperparameter optimization.
  * The figure below shows a summary of the VAE results, also using the FiberCup dataset:
.. image:: https://github.com/dipy/dipy.org/blob/master/_static/images/gsoc/2024/inigo/vanilla_vae_120_epoch_results.png
  :alt: VAE architecture results on the FiberCup dataset.
  :width: 800


* Implemented a conditional Variational Autoencoder (condVAE) architecture based on the `Variational AutoEncoders for Regression <https://doi.org/10.1007/978-3-030-32245-8_91>`_ paper. The model is found in the ``cond_vae_model.py`` module of the `TractoEncoder GSoC <https://github.com/itellaetxe/tractoencoder_gsoc>`_ repository. The model was trained on the FiberCup dataset, and the conditioning variable in this case was chosen to be the length of the streamlines, hypothesizing that this is a relatively simple feature to capture by the model based on their geometry. and the results are shown in the figure below:

* Implemented validation strategies of the condVAE model to check that the model can capture the variability of the conditioning variable

.. * Worked on `CC359 <https://sites.google.com/view/calgary-campinas-dataset/home>`_ & `NFBS <http://preprocessed-connectomes-project.org/NFB_skullstripped/>`_ datasets, both consist of T1-weighted human brain MRI with 359 & 125 samples respectively. Preprocessed each input volume following the 3 steps below-

..   * Skull-stripping the dataset, if required, using existing masks.
..   * Pre-process using ``transform_img`` function - perform voxel resizing & affine transformation to obtain final (128,128,128,1) shape & (1,1,1) voxel shape
..   * Neutralized background pixels to 0 using respective masks
..   * MinMax normalization to rescale intensities to (0,1) 

.. * Implemented 3D versions of the above repositories from scratch

..   * VQVAE3D

..     * The encoder & decoder of 3D VQVAE are symmetrical with 3 Convolutional & 3 Transpose Convolutional layers respectively, followed by non-linear ``relu`` units
..     * Vector Quantizer trains a learnable embedding matrix to identify closest latents for a given input based on L2 loss function
..     * VQVAE gave superior results over VAE as shown in `this <https://arxiv.org/pdf/1711.00937.pdf>`_ paper, owing to the fact that quantizer addresses the problem of 'Posterior Collapse' seen in traditional VAEs
..     * Trained the model for approximately 100 epochs using Adam optimizer with lr=1e-4, minimized reconstruction & quantizer losses together
..     * Test dataset reconstructions-
    
..     .. image:: https://github.com/dipy/dipy/blob/master/doc/_static/vqvae3d-reconst-f3.png
..       :alt: VQVAE reconstructions on NFBS test dataset
..       :width: 800

..   * 3D LDM

..     * Built unconditional Latent Diffusion Model(LDM) combining `DDPM <https://arxiv.org/pdf/2006.11239.pdf>`_ & `Stable Diffusion <https://arxiv.org/pdf/2112.10752.pdf>`_ implementations
..     * U-Net of the reverse process consists of 3 downsampling & 3 upsampling layers each consisting of 2 residual layers and an optional attention layer
..     * Trained the model using linear (forward)variance scaling & various diffusion steps - 200, 300
..     * Adopted `algorithm 4 <https://arxiv.org/pdf/2006.11239.pdf>`_ for sampling synthetic generations at 200 & 300 diffusion steps-
..     .. image:: https://github.com/dipy/dipy/blob/master/doc/_static/dm3d-reconst-D200-D300.png
..        :alt: 3D LDM synthetic generations
..        :width: 800


.. * Adopted MONAI's implementation

..   * Replaced VQVAE encoder & decoder with a slightly complex architecture that includes residual connections alternating between convolutions
..   * Carried out experiments with same training parameters with varying batch sizes & also used both datasets in a single experiment


..     .. image:: https://github.com/lb-97/dipy/blob/blog_branch_week_12_13/doc/_static/vqvae3d-monai-training-plots.png
..        :alt: VQVAE-MONAI training plots
..        :width: 800
     
  
..   * Clearly the training curves show that the higher batch size & dataset length, the better the stability of the training metric for learning rate=1e-4
..   * Plotted reconstructions for top two experiments - (Batch size=12, Both datasets) & (Batch size=5, NFBS dataset)


..     .. image:: https://github.com/lb-97/dipy/blob/blog_branch_week_12_13/doc/_static/vqvae-reconstructions-comparison.png
..        :alt: VQVAE-MONAI reconstructions on best performing models
..        :width: 800
  
..   * Existing diffusion model has been trained on these new latents to check for their efficacy on synthetic image generation
..   * The training curves converged pretty quickly, but the sampled generations are still pure noise

..     .. image:: https://github.com/lb-97/dipy/blob/blog_branch_week_12_13/doc/_static/dm3d-monai-training-curves.png
..        :alt: 3D LDM training curve for various batch sizes & diffusion steps
..        :width: 400
..   * To summarize, we've stretched the capability of our VQVAE model despite being less complex with only ``num_res_channels=(32, 64)``. We consistently achieved improved reconstruction results with every experiment. Our latest experiments are trained using a weighted loss function with lesser weight attached to background pixels owing to their higher number. This led to not just capturing the outer structure of a human brain but also the volumetric details resembling microstructural information inside the brain. This is a major improvement from all previous trainings.

..   * For future work we should look into two things - debugging Diffusion Model, scaling VQVAE model.

..     * As a first priority, we could analyze the reason for pure noise output in DM3D generations, this would help us rule out any implementation errors of the sampling process.

..     * As a second step, we could also try scaling up both VQVAE as well as the Diffusion Model in terms of complexity, such as increasing intermediate channel dimensions from 64 to 128 or 256. This hopefully may help us achieve the state-of-art on NFBS & CC359 datasets.


.. Objectives in Progress
.. ----------------------

.. * Unconditional LDM hasn't shown any progress in generations yet. Increasing model complexity with larger number of intermediate channels & increasing diffusion steps to 1000 is a direction of improvement
.. * Implemented cross-attention module as part of U-Net, to accommodate conditional training such as tumor type, tumor location, brain age etc
.. * Implementation of evaluation metrics such as FID(Frechet Inception Distance) & IS(Inception Score) will be useful in estimating the generative capabilities of our models


.. Timeline
.. --------

.. .. list-table::
..    :header-rows: 1

..    * - Date
..      - Description
..      - Blog Post Link
..    * - Week 0\  :raw-html:`<br>`\ (19-05-2023)
..      - Journey of GSOC application & acceptance
..      - `DIPY <https://github.com/dipy/dipy/blob/master/doc/posts/2023/2023_05_19_vara_week0.rst>`_
..    * - Week 1\  :raw-html:`<br>`\ (29-05-2023)
..      - Community bonding and Project kickstart
..      - `DIPY <https://github.com/dipy/dipy/blob/master/doc/posts/2023/2023_05_29_vara_week1.rst>`_
..    * - Week 2\  :raw-html:`<br>`\ (05-06-2023)
..      - Deep Dive into VQVAE
..      - `DIPY <https://github.com/dipy/dipy/blob/master/doc/posts/2023/2023_06_05_vara_week2.rst>`_
..    * - Week 3\  :raw-html:`<br>`\ (12-06-2023)
..      - VQVAE results and study on Diffusion models
..      - `DIPY <https://github.com/dipy/dipy/blob/master/doc/posts/2023/2023_06_12_vara_week3.rst>`_
..    * - Week 4\  :raw-html:`<br>`\ (19-06-2023)
..      - Diffusion research continues
..      - `DIPY <https://github.com/dipy/dipy/blob/master/doc/posts/2023/2023_06_19_vara_week4.rst>`_
..    * - Week 5\  :raw-html:`<br>`\ (26-06-2023)
..      - Carbonate HPC Account Setup, Experiment, Debug and Repeat
..      - `DIPY <https://github.com/dipy/dipy/blob/master/doc/posts/2023/2023_06_26_vara_week5.rstt>`_
..    * - Week 6 & Week 7\  :raw-html:`<br>`\ (10-07-2023)
..      - Diffusion Model results on pre-trained VQVAE latents of NFBS MRI Dataset
..      - `DIPY <https://github.com/dipy/dipy/blob/master/doc/posts/2023/2023_07_10_vara_week6_and_week7.rst>`_
..    * - Week 8 & Week 9\  :raw-html:`<br>`\ (24-07-2023)
..      - VQVAE MONAI models & checkerboard artifacts
..      - `DIPY <https://github.com/dipy/dipy/blob/master/doc/posts/2023/2023_07_24_vara_week_8_9.rst>`_
..    * - Week 10 & Week 11\  :raw-html:`<br>`\ (07-08-2023)
..      - HPC issues, GPU availability, Tensorflow errors: Week 10 & Week 11
..      - `DIPY <https://github.com/dipy/dipy/blob/master/doc/posts/2023/2023_08_07_vara_week_10_11.rst>`_
..    * - Week 12 & Week 13\  :raw-html:`<br>`\ (21-08-2023)
..      - Finalized experiments using both datasets
..      - `DIPY <https://github.com/dipy/dipy/blob/master/doc/posts/2023/2023_08_21_vara_week_12_13.rst>`_