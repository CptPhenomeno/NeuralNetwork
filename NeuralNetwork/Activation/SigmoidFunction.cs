namespace NeuralNetwork.Activation
{
    using System;

    public class SigmoidFunction : IActivationFunction
    {
        //Params of function
        private double alpha;

        public SigmoidFunction(double alpha = 1.0)
        {
            this.alpha = alpha;
        }

        public double Function (double x)
        {
            return 1.0/(1.0 + Math.Exp(-alpha * x));
        }

        public double Derivative (double x)
        {
            double y = Function(x);
            return alpha * y * (1.0 - y);
        }
    }
}
