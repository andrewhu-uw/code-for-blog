# Visual Studio

I got this to compile on Visual Studio 2015, but I did not make a CMake configuration
I'm only really uploading my VS files so that I can test this again on 
my other machine. If you're wondering about configuring it for VS,
I'm not really the guy to talk to, but you can shoot me an email if
you want (edu uw at andrewhu)

Also, I couldn't get the dump_memory function to work on Windows, but
it does work on Linux

# Usage

Right now, you run `echo -e "[INPUT STRING]\0 [INDEX]" | x64/Debug/llvm-jit.exe echo_plus.bf"`

And then it should print out that string with each character shifted up one on the ASCII table,

Then it should print out the character in the memory index you selected followed by the '!' character
