using System;

class Program
{
    static void Main()
    {
        var line = Console.ReadLine() ?? string.Empty;
        var parts = line.Split(new[] {' ', '\t'}, StringSplitOptions.RemoveEmptyEntries);
        long a = 0, b = 0;
        if (parts.Length >= 2)
        {
            long.TryParse(parts[0], out a);
            long.TryParse(parts[1], out b);
        }
        Console.WriteLine(a + b);
    }
}

