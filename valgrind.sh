export VALGRIND_LIB=/usr/local/lib/valgrind
valgrind --leak-check=full --track-origins=yes --suppressions=/usr/share/openmpi/openmpi-valgrind.supp build/oscillator $@
#--tool=exp-sgcheck
