namespace NeuralNetwork.Activation
{
    public class Threshold : IActivationFunction
    {
        private double threshold;

        public Threshold(double thresholdValue = 0.5)
        {
            threshold = thresholdValue;
        }

        public double Function(double x)
        {
            return (x >= threshold) ? 1.0 : 0.0;
        }

        public double Derivative(double x)
        {
            throw new System.NotImplementedException();
        }
    }
}
