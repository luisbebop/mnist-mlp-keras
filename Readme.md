# mnist-mlp-keras (Deep learning intro)

This project teaches you how to use `machine learning` to recognize handwriteen digits. Consider this the `Hello World` of `AI`.

I decided to use `javascript` and `microservices` to deploy this project, because it makes easy to run examples, without the need of deploying servers or learn a new programming language.

This network recognizes digits using a `multilayer perceptron`. I strongly recommend you to watch the videos bellow to understand the mathematical concepts behind this algorithm:

[But what *is* a Neural Network? | Deep learning, chapter 1](https://www.youtube.com/watch?v=aircAruvnKk)

[Gradient descent, how neural networks learn | Deep learning, chapter 2](https://www.youtube.com/watch?v=IHZwWFHWa-w)

[What is backpropagation and what is it actually doing? | Deep learning, chapter 3](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

[Backpropagation calculus | Appendix to deep learning chapter 3](https://www.youtube.com/watch?v=tIeHLnjs5U8)

## Details

I'm using `Javascript` and `AWS lambda`. The interface is based on [this tutorial](https://github.com/llSourcell/how_to_deploy_a_keras_model_to_production).

It uses [Keras](https://github.com/fchollet/keras) to train the network and [AWS lambda](https://aws.amazon.com/lambda/) to serve the API endpoint using [Keras-Js](https://github.com/transcranial/keras-js) and [Node](https://nodejs.org)

I decided to use `Keras-Js` instead of `flask+python`, because it opens the `possibility` put the model on the `client side` and it's easier to create a `serverless` prediction endpoint API.

## Demo

[Live demo](https://psl253vifh.execute-api.us-west-2.amazonaws.com/development/)

## Setup

```
# install tensorflow
# install keras
$ npm install
```

## Train new model

```
$ python train.py
$ python encoder.py mnist.hdf5
```

## Test

```
$ node mnist-serve.js
```

## Deploy

Install [Up](https://github.com/apex/up) 

```
$ up
```

## Notes

The `build` hook in `up.json` simply runs `make`, which ensures that the `./node-v8.4.0-linux-x64` binary is downloaded and present, Make is used so that this process only happens once.

```json
{
  "hooks": {
    "build": "make"
  }
}
```

The `proxy.command` script is run inside Lambda to start your server. You can think of this as `npm start`, however you'd likely want `npm start` to be `node app.js` for local development. Defining `proxy.command` is strictly for running in production.

```json
{
  "proxy": {
    "command": "./node-v8.7.0-linux-x64/bin/node app.js"
  }
}
```

Also note that `./node-v8.7.0-linux-x64` is placed in .gitignore so it's not checked into GIT. Up will ignore these by default, so we negate it with `!node-v8.7.0-linux-x64` in .upignore.
