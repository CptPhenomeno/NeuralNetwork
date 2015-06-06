namespace NeuralNetwork.Layer
{
    public class LayerFactory
    {

        public static Layer Clone(Layer toClone)
        {
            Layer l = new Layer(toClone.ActivationFunction, toClone.NumberOfNeurons, toClone.NumberOfInputs);
            l.Weights = toClone.Weights.Clone();
            l.Bias = toClone.Bias.Clone();

            return l;
        }
    }
}
