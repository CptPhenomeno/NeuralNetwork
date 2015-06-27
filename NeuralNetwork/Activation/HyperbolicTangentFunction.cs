namespace NeuralNetwork.Activation
{
    using System;

    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;

    public class HyperbolicTangentFunction : IActivationFunction
    {
        private double alpha;
        private double beta;

        public HyperbolicTangentFunction(double alpha = 1, double beta = 1)
        {
            this.alpha = alpha;
            this.beta = beta;
        }

        public Vector<double> Function(Vector<double> x)
        {
            return alpha * x.Map(e => Math.Tanh(beta * e));
        }

        public Vector<double> Derivative(Vector<double> x)
        {
            Vector<double> y = Function(x);
            return (beta / alpha) * y.Map(e => (alpha - e) * (alpha + e));
        }
    }
}
