#!/usr/bin/env bash

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e +x

if [ -z ${BRANCH} ]; then
    BRANCH="paddle_compiler"
fi

function init() {
    PADDLE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../../../../" && pwd )"
    export PADDLE_ROOT
    if [ -z "${SCRIPT_NAME}" ]; then
        SCRIPT_NAME=$0
    fi

    ENABLE_MAKE_CLEAN=${ENABLE_MAKE_CLEAN:-ON}

    # NOTE(chenweihang): For easy debugging, CI displays the C++ error stacktrace by default
    export FLAGS_call_stack_level=2

    # set CI_SKIP_CPP_TEST if only *.py changed
    # In order to avoid using in some CI(such as daily performance), the current
    # branch must not be `${BRANCH}` which is usually develop.
    if [ ${CI_SKIP_CPP_TEST:-ON} == "OFF"  ];then
        echo "CI_SKIP_CPP_TEST=OFF"
    else
        if [ "$(git branch | grep "^\*" | awk '{print $2}')" != "${BRANCH}" ]; then
            git diff --name-only ${BRANCH} | grep -v "\.py$" || export CI_SKIP_CPP_TEST=ON
        fi
    fi
}

function parallel_test() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    pip install ${PADDLE_ROOT}/build/python/dist/*whl
    cp ${PADDLE_ROOT}/build/python/paddle/fluid/tests/unittests/op_test.py ${PADDLE_ROOT}/build/python
    ut_total_startTime_s=`date +%s`

    EXIT_CODE=0

    all_ctests=$(ctest -N | awk -F ': ' '{print $2}' | sed '/^$/d' | sed '$d')
    compiler_files_contains_test=$(find ${PADDLE_ROOT}/paddle/fluid/compiler -iname "*test*" -type f)

    # Temporary directory to save failed Piano test
    tmp_dir=`mktemp -d`
    echo "Created temporary directory to store test result:" $tmp_dir

    piano_test_list=''
    for test_case in $all_ctests; do
	is_piano_test=0
	for test_file in $compiler_files_contains_test; do
	    if [[ $test_file =~ $test_case ]]; then
                is_piano_test=1
                break
   	    fi
	done

        if [ $is_piano_test = 1 ]; then
            tmp_file_rand=`date +%s%N`
            tmp_file=$tmp_dir/$tmp_file_rand
	    ctest -R $test_case --output-on-failure | tee $tmp_file
	    piano_test_list="${piano_test_list}
	    ${test_case}"
	fi
    done

    failed_test_lists=''
    set +e
    for file in `ls $tmp_dir`; do
        grep -q 'The following tests FAILED:' $tmp_dir/$file
        grep_exit_code=$?
        if [ $grep_exit_code -ne 0 ]; then
            failed_test=''
        else
	    EXIT_CODE=8
            failed_test=`grep -A 10000 'The following tests FAILED:' $tmp_dir/$file | sed 's/The following tests FAILED://g'|sed '/^$/d'`
            failed_test_lists="${failed_test_lists}
            ${failed_test}"
        fi
    done
    set -e

    rm -rf $tmp_dir
    echo "Removed temporary directory which stores test result:" $tmp_dir

    if [ $EXIT_CODE != 0 ]; then
	echo "========================================="
        echo "The following tests FAILED:"
	echo "========================================="
        echo "${failed_test_lists}"
        exit 8;
    fi

    ut_total_endTime_s=`date +%s`
    echo "========================================="
    echo "The following tests SUCCESSED:"
    echo "========================================="
    echo "${piano_test_list}"
    echo ""
    echo "TestCases Total Time: $[ $ut_total_endTime_s - $ut_total_startTime_s ]s"
    echo "ipipe_log_param_TestCases_Total_Time: $[ $ut_total_endTime_s - $ut_total_startTime_s ]s"
}

function main() {
    init
    parallel_test

    echo "piano_ci_test script finished as expected"
}

main $@
