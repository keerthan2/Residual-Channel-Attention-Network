#!/bin/bash
matlab17b -nodisplay -nojvm -nosplash -nodesktop -r "try, run('evaluate_scoreY.m'), catch, exit(1), end, exit(0);"
echo "matlab exit code: $?"
