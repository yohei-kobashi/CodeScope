use strict;
use warnings;

my $line = <STDIN>;
my ($a, $b) = split ' ', $line // '';
$a //= 0; $b //= 0;
print $a + $b, "\n";

