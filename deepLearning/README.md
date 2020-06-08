# Overview 

Contained are some deep learning models.  I typically train with [https://pytorch.org/](PyTorch) then transition to production with the CXX backend.

## Three Component Pickers

Three-component U-Net architecture pickers are trained for both Utah and Yellowstone.  Since we have far fewer picks in our database than California I prefer to use picks from both regions but use only Utah noise in Utah and Yellowstone noise in Yellowstone.

### Yellowstone Picker

This is the Yellowstone three-component picker.  The primary challenge in Yellowstone is data-completeness and the resulting difficulty in harvesting noise.

### Utah Picker

This is the Utah three-component picker.  While data-completeness in Utah is excellent the primary challenges in gathering noise are quarry blasts.  Additionally, stations near the Book Cliffs regions are excluded from training since there appears to be difficulty in distinguishing anthropogenic and natural events.

The Utah picker has one additional perturbation for the Magna sequence.  In this case all Magna events are not relegated to the testing dataset.

## First Motion Pickers

This is a single (vertical) channel polarity picker trained on both Yellowstone and Utah data.

