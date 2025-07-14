using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Linq.Expressions;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using System.Threading;
using Newtonsoft.Json;

namespace Action_alg
{
    [Serializable]
    public class MyDataType
    {
        public int Loc { get; set; }  //目标编号
        public float Ret_time { get; set; }  //返回基地时刻
        public float Rec_time { get; set; }  //打击目标时刻
        public Expression<Func<float, float, float, float, float, float, float, float, float>> Lambda { get; set; }  //能力分配实例化
        public Func<float, float, float, float, float, float, float, float, float> CompiledLambda { get; set; }

        public MyDataType(int intValue, float floatValue1, float floatValue2, Expression<Func<float, float, float, float, float, float, float, float, float>> lambda)
        {
            Loc = intValue;
            Ret_time = floatValue1;
            Rec_time = floatValue2;
            Lambda = lambda;
            CompiledLambda = lambda.Compile();
        }
    }
    public class MyDataType_return
    {
        public int Loc { get; set; }  //目标编号
        public float Ret_time { get; set; }  //返回基地时刻
        public float Rec_time { get; set; }  //打击目标时刻
        public float u { get; set; }  //能力分配占比

        public MyDataType_return(int intValue, float floatValue1, float floatValue2, float floatValue3)
        {
            Loc = intValue;
            Ret_time = floatValue1;
            Rec_time = floatValue2;
            u = floatValue3;
        }
    }
    public class Program
    {
        static List<MyDataType> Alg(int P, float[] Time)
        {
            List<ParameterExpression> U = new List<ParameterExpression>();
            for (int i = 1; i <= P; i++)
            {
                string paramName = "u" + i;
                ParameterExpression param = Expression.Parameter(typeof(float), paramName);
                U.Add(param);
            }

            List<MyDataType> Action_Lis = new List<MyDataType>();
            for (int i = 0; i < P; i++)
            {
                var func = Expression.Lambda<Func<float, float, float, float, float, float, float, float, float>>(U[i], U[0], U[1], U[2], U[3], U[4], U[5], U[6], U[7]);
                Action_Lis.Add(new MyDataType(i + 1, Time[i] * 2, Time[i], func));
            }

            List<MyDataType> Action_Lis_confirm = new List<MyDataType>();
            while (true)
            {
                Action_Lis = Action_Lis.OrderBy(x => x.Ret_time).ToList();
                if (Action_Lis[0].Ret_time > (3.0f + 23.665621261737517f / 38.304951684997060f)) break;
                MyDataType act = Action_Lis[0];

                Action_Lis_confirm.Add(act);
                Action_Lis.RemoveAt(0);
                for (int i = 0; i < P; i++)
                {
                    Expression body = Expression.Multiply(act.Lambda.Body, U[i]);
                    var newFuncExpr = Expression.Lambda<Func<float, float, float, float, float, float, float, float, float>>(body, U[0], U[1], U[2], U[3], U[4], U[5], U[6], U[7]);
                    Action_Lis.Add(new MyDataType(i + 1, act.Ret_time + Time[i] * 2, act.Ret_time + Time[i], newFuncExpr));
                }
            }
            Action_Lis_confirm = Action_Lis_confirm.OrderBy(x => x.Ret_time).ToList();

            // 创建一个新列表来存储合并后的结果
            List<MyDataType> Merged_Action_Lis = new List<MyDataType>();
            bool found = false;

            for (int i = 0; i < Action_Lis_confirm.Count; ++i)
            {
                found = false;
                MyDataType item = Action_Lis_confirm[i];
                for (int j = 0; j < Merged_Action_Lis.Count; ++j)
                {
                    if (Math.Abs(item.Ret_time - Merged_Action_Lis[j].Ret_time) <= 0.05f && item.Loc == Merged_Action_Lis[j].Loc)
                    {
                        Expression body = Merged_Action_Lis[j].Lambda.Body;
                        body = Expression.Add(body, Action_Lis_confirm[i].Lambda.Body);
                        var newFuncExpr = Expression.Lambda<Func<float, float, float, float, float, float, float, float, float>>(body, U[0], U[1], U[2], U[3], U[4], U[5], U[6], U[7]);
                        Merged_Action_Lis[j] = new MyDataType(Merged_Action_Lis[j].Loc, Merged_Action_Lis[j].Ret_time, Merged_Action_Lis[j].Rec_time, newFuncExpr);

                        found = true;
                        break;
                    }
                }

                if (!found) Merged_Action_Lis.Add(item);
            }
            return Merged_Action_Lis;
        }
        static List<float> Num(List<MyDataType> Merged_Action_Lis, float[] pos, float N)
        {
            List<MyDataType_return> Merged_Action_Lis_return = new List<MyDataType_return>();
            foreach (var item in Merged_Action_Lis)
            {
                Func<float, float, float, float, float, float, float, float, float> compiledLambda = item.Lambda.Compile();
                Merged_Action_Lis_return.Add(new MyDataType_return(item.Loc, item.Ret_time, item.Rec_time,
                                                                                                               compiledLambda(pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6], pos[7]) * N));
            }
            Merged_Action_Lis_return = Merged_Action_Lis_return.OrderBy(x => x.Rec_time).ToList();

            List<float> Merged_Action_Lis_return_num = new List<float>();
            foreach (var item in Merged_Action_Lis_return)
            {
                Merged_Action_Lis_return_num.Add(item.u);
            }
            return Merged_Action_Lis_return_num;
        }

        public static void SaveToTxt(string filePath, List<MyDataType> objects)
        {
            using (var writer = new StreamWriter(filePath))
            {
                foreach (var obj in objects)
                {
                    writer.WriteLine($"Loc: {obj.Loc}");
                    writer.WriteLine($"Ret_time: {obj.Ret_time}");
                    writer.WriteLine($"Rec_time: {obj.Rec_time}");
                    writer.WriteLine($"Lambda: {obj.Lambda}");
                    writer.WriteLine();
                }
            }
        }

        /// 任务总数
        public int DataCount { get; set; }
        /// GroupDataCount个时间完成任务
        public int GroupDataCount { get; set; }
        /// 任务结果
        public List<List<float>> Results { get; set; }
        public Program(int dataCount, int groupDataCount)
        {
            DataCount = dataCount;
            GroupDataCount = groupDataCount;
            this.Results = new List<List<float>>();
        }


        /// <summary>
        /// 总数据处理
        /// </summary>
        public void OverallDataProcessing(List<MyDataType> Merged_Action_Lis, List<float[]> Pos, float N, float[] x0, float symbol, int Search_num)
        {
            List<int> data = Enumerable.Range(1, DataCount).ToList();
            int groupCount = DataCount % GroupDataCount == 0 ? DataCount / GroupDataCount : DataCount / GroupDataCount + 1;
            Task[] tasks = new Task[groupCount];
            for (int i = 0; i < Search_num; i++)
            {
                this.Results.Add(null);
                int index = i;
                tasks[index] = Task.Run(() => ProcessAsync(index, Merged_Action_Lis, Pos[index], N, x0, symbol)); //代数式，狼状态 （动作实例化）   初始状态，结束标志
            }
            Task.WaitAll(tasks);
        }

        public void ProcessAsync(int index, List<MyDataType> Merged_Action_Lis, float[] pos, float N, float[] x0, float symbol)
        {
            this.Results[index] = Num(Merged_Action_Lis, pos, N);
        }

        static void Main(string[] args)
        {
            int Search_num;
            int P;
            float N;
            if (!int.TryParse(args[0], out Search_num) || !int.TryParse(args[1], out P) || !float.TryParse(args[2], out N))
            {
                Console.WriteLine("Invalid input for Search_wolf or P or N.");
                return;
            }

            float[] Time = new float[P];
            for (int i = 3; i < 3 + P; i++)
            {
                if (!float.TryParse(args[i], out Time[i - 3]))
                {
                    Console.WriteLine($"Invalid input for Time at index {i - 3}.");
                    return;
                }
            }

            float[] x0 = new float[P];
            for (int i = 3 + P; i < 3 + 2 * P; i++)
            {
                if (!float.TryParse(args[i], out x0[i - 3 - P]))
                {
                    Console.WriteLine($"Invalid input for x0 at index {i - 3 - P}.");
                    return;
                }
            }

            float symbol;
            if (!float.TryParse(args[3 + 2 * P], out symbol))
            {
                Console.WriteLine("Invalid input for symbol.");
                return;
            }

            List<float[]> Pos = new List<float[]>();
            for (int i = 0; i < Search_num; ++i)
            {
                float[] pos = new float[P];
                for (int j = 0; j < P; ++j)
                {
                    if (!float.TryParse(args[4 + 2 * P + i * P + j], out pos[j]))
                    {
                        Console.WriteLine("Invalid input for pos.");
                        return;
                    }
                }
                Pos.Add(pos);
            }

            List<MyDataType> Merged_Action_Lis = Alg(P, Time);

            Program test = new Program(Search_num, 1);
            test.OverallDataProcessing(Merged_Action_Lis, Pos, N, x0, symbol, Search_num);
            // 序列化为JSON字符串  
            string json = JsonConvert.SerializeObject(test.Results, Formatting.Indented);
            // 输出到控制台  
            Console.WriteLine(json);




            //SaveToTxt(@"D:\serializeMyDataType.txt", Merged_Action_Lis);
            /*            FileStream fs = new FileStream(@"D:\serializeMyDataType.dat", FileMode.Create);
                        BinaryFormatter bf = new BinaryFormatter();
                        bf.Serialize(fs, Merged_Action_Lis);
                        fs.Close();*/
        }
    }
}
