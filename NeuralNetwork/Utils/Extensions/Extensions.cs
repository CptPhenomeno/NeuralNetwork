namespace NeuralNetwork.Utils.Extensions
{
    using System;

    public static class ArrayExtensions
    {
        public static string Print(this Array array)
        {
            string s;
            if (array.Length > 0)
            {
                int i;
                object value;
                s = "[";
                for (i = 0; i < array.Length - 1; i++)
                {
                    value = array.GetValue(i);
                    s += (value is Array) ? ((Array)value).Print() + ",\n " : value + ",";
                }


                value = array.GetValue(i);
                s += (value is Array) ? ((Array)value).Print() + "]" : value + "]";
            }
            else
                s = "[]";
            return s;
        }

        public static void Swap(this Array array, int pos1, int pos2)
        {
            object o1 = array.GetValue(pos1);
            object o2 = array.GetValue(pos2);
            array.SetValue(o2, pos1);
            array.SetValue(o1, pos2);
        }

        public static void Shuffle(this Array array)
        {
            int length = array.Length;
            Random rand = new Random();
            for (int pos = length - 1; pos > 0; pos--)
            {
                int r = rand.Next(0, pos);
                array.Swap(r, pos);
            }
        }

        public static void TwinShuffle(Array arr1, Array arr2)
        {
            int length = arr1.Length;
            Random rand = new Random();
            for (int pos = length - 1; pos > 0; pos--)
            {
                int r = rand.Next(0, pos);
                arr1.Swap(r, pos);
                arr2.Swap(r, pos);
            }
        }
    }

}
