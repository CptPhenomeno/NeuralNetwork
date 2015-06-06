using NeuralNetwork.Activation;
using NeuralNetwork.Layer;

namespace NeuralNetwork.Network
{
    public class NetworkFactory
    {
        public static NeuralNet Clone(NeuralNet toClone)
        {
            int[] sizeOfLayers = new int[toClone.NumberOfLayers];
            IActivationFunction[] activationFunctionOFLayers = new IActivationFunction[toClone.NumberOfLayers];

            for (int layerIndex = 0; layerIndex < toClone.NumberOfLayers; layerIndex++)
            {
                sizeOfLayers[layerIndex] = toClone[layerIndex].NumberOfNeurons;
                activationFunctionOFLayers[layerIndex] = toClone[layerIndex].ActivationFunction;
            }
            NeuralNet n = new NeuralNet(toClone.NumberOfInputs, sizeOfLayers, activationFunctionOFLayers);

            for (int layerIndex = 0; layerIndex < toClone.NumberOfLayers; layerIndex++)
            {
                n[layerIndex] = null;
                n[layerIndex] = LayerFactory.Clone(toClone[layerIndex]);
            }

            return n;
        }
    }
}
