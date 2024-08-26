# Llava-colab

A demo of using the Llava model to caption images in Gradio.

First, run the backend in the **Llava_colab** notebook by opening it in Colab.. Open public gradio URL (e.g. something like https://8b5441fa7340c4e8fc.gradio.live).  Then drag an image into the box and click to get a caption.  If you leave the prompt and temperature fields blank, the labeling will be run using the default values.

If you're a programmer and would like to manually caption your images, make sure you've run the backend above and noted the public gradio URL.

Then open the **Llava_api** notebook and paste the public gradio URL into the **gradio_url** field at the top to run the two examples.
