namespace NeuralNetwork.Activation
{
    using System;

    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;
    
    public class SigmoidFunction : IActivationFunction
    {
        //Params of function
        private double alpha;

        public SigmoidFunction(double alpha = 1.0)
        {
            this.alpha = alpha;
        }

        public Vector<double> Function(Vector<double> x)
        {
            return x.Map(e => 1.0 / (1.0 + Math.Exp(-alpha * e)));
        }

        public Vector<double> Derivative(Vector<double> x)
        {
            Vector<double> y = Function(x);
            return alpha * (y.Map(e => e * (1.0 - e)));
        }
    }
}
