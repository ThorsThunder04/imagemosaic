
# Image Mosaic

This program should make a mosaic image based off of a target image and a given dataset of other images.

So for each pixel of the target image, we try and find one of the dataset images that matches the color of that pixel the best. Then we place the image at that location.
In the end, we end up getting a mosaiced image made up of loads of tile images.

# Dependencies

You just need Python 3.10 or later. But for the notebook, you should have anaconda to run it. Or you can find some other means of running it, like [google colab](https://colab.research.google.com/) for example.

Install the required libraries with:

```pip install -r requirements.txt```

# Why make this?

Back when I was generating AI images, I had this idea to make a massive image. Something to make me stand out to some extent (other then me currating notebooks for other none tech savvy artists). Unfortunately I stopped being so serious about AI generation before I could finish it. Mostly because of google colab's payment plan changing for powerful GPUs.

The Idea of it is that I can generate one target image, and then based off of loads of images I had previously generated (using CC12M at the time) I could make a massive mosaiced image with matched colors. So it would be the biggest AI generated image composed of AI generated images. So at a resolution of 256x256 for example, it would be an AI generated image formed from 65536 AI generated images. Which will ofc make a 65536x65536 image.

Reaching such a large number of generated images seems hard, but I was doing it pretty quickly. CC12M was the fastest generator at the time I'm pretty sure. The Stable Diffusion came around of course. But didn't start working on it with SD yet (but it probably would have yielded nicer results).

Of course there were existing programs to do this already. But they were either payed, or they didn't use the dataset images in the way I liked (instead of picking an image close to a pixel color, it picks an image and then just modifies it's color). And I wasn't gonna pay for something that I was maybe just gonna use once.
I also checked out some github projects of it, but unfortunately they weren't what I wanted either. Some of them didn't keep the original resolution of the source dataset images (which is what I wanted). I wanted you to be able to zoom into the big image, and see each individual tile in full detail if you wanted.

So in the end, I set out to program this myself. I must improve on it, since it is not super customizable or optimized currently.
