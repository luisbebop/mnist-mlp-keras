const express = require('express')
const bodyparser = require('body-parser')
const fs = require('fs')
const KerasJS = require('keras-js')

const app = express()
const { PORT = 3000 } = process.env

app.use(bodyparser.json())
app.use(express.static('public'))

const model = new KerasJS.Model({
  filepaths: {
    model: 'mnist.json',
    weights: 'mnist_weights.buf',
    metadata: 'mnist_metadata.json'
  },
  filesystem: true
})

async function loadModel() {
  try {
    await model.ready()
  } catch (err) {
    console.log('model error: ' + err)
  }
}

app.get('/', function(req, res){
  res.redirect("mnist.html")
})

app.post('/checkNumber', async (req, res, next) => {
  var inputData = {
    input: new Float32Array(req.body.input)
  }
  
  const outputData = await model.predict(inputData)
  var max = outputData.output.reduce((max, activation) => Math.max(max, activation), 0)
  var guess = outputData.output.indexOf(max)
  res.send(guess.toString())
})

console.log('loading model')
loadModel();
console.log('model loaded')
console.log('listening on %s', PORT)
app.listen(PORT)