import sys
from MI_Classes import VAE
import keras

image_folder = '/n/groups/patel/Alan/Aging/Medical_Images/images/Musculoskeletal/FullBody/Mixed/'
nb_images = 1

path_images = '/n/groups/patel/Alan/Aging/Medical_Images/images/Musculoskeletal/FullBody/'
input_shape = (541, 181, 3)
batch_size = 64
latent_dim = 256
optimizer = keras.optimizers.Adam()
nb_epochs=30


vae = VAE(input_shape, batch_size=batch_size , latent_dim=latent_dim)
vae.load_data(path_images=path_images, class_name='Mixed')
vae.create_encoder(logs=True)
vae.create_decoder(logs=True)
vae.fit_images(optimizer, nb_epochs=nb_epochs)
vae.proccess_and_save(nb_images, True, False)

    

# vae = VAE(input_shape, batch_size=batch_size , latent_dim=10, debug_mode=True)
# vae.load_data(nb_images=nb_images, random=True)
# vae.create_encoder(logs=True)
# vae.create_decoder(logs=True)
# vae.fit_images(optimizer, nb_epochs=nb_epochs)
# vae.proccess_and_save(nb_images, True, False)