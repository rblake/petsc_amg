
include ${PETSC_DIR}/bmake/common/variables

OBJS += mglib.o

TESTS += multigrid_test1
TESTS += multigrid_test2

all: ${TESTS}
clean:
	rm -f *.o ${TESTS}

%: %.o ${OBJS}
	${PCC} ${PCC_LINKER_FLAGS} -o $@ $^ ${PETSC_KSP_LIB}

%.o: %.cc
	${CXX} -c ${PCC_FLAGS} ${COPTFLAGS} ${CFLAGS} ${CCPPFLAGS} $(FLAGS) -c  -o $@ $<

%.o: %.c  
	${CXX} -c ${PCC_FLAGS} ${COPTFLAGS} ${CFLAGS} ${CCPPFLAGS} $(FLAGS) -c  -o $@ $<

%: %.c #Disable default makefile rule.
%: %.cc 

.PRECIOUS: $(OBJS) $(TESTS:%=%.o)