import std.stdio;
import std.string : split;
import std.conv : to;

void main()
{
    auto line = readln();
    auto parts = line.split();
    if (parts.length >= 2)
    {
        long a = to!long(parts[0]);
        long b = to!long(parts[1]);
        writeln(a + b);
    }
}

