#!/bin/bash

shopt -s globstar

rm -f .coverage  # can conflict with tox


echo -n "Upgrade HFST and vislcg3 (root password may be required)? (y/n) "
read answer
if [ "$answer" != "${answer#[Yy]}" ] ; then
    case "$(uname -s)" in
        Darwin)
            curl https://apertium.projectjj.com/osx/install-nightly.sh | sudo bash
            ;;

        Linux)
	    curl -sS https://apertium.projectjj.com/apt/install-nightly.sh | sudo bash
	    sudo apt-get -f install apertium-all-dev
            ;;

        CYGWIN*|MINGW32*|MSYS*|MINGW*)
            echo 'MS Windows, HFST will not be installed.'
            ;;

        *)
            echo 'Error: Unknown OS, HFST will not be installed.'
            ;;
    esac
else
    echo "Not upgrading hfst and vislcg3."
fi


echo "Versions with which tests passed for this commit:" \
    > hfst_vislcg3_versions.txt
hfst-tokenize --version | grep hfst >> hfst_vislcg3_versions.txt
vislcg3 --version | grep VISL >> hfst_vislcg3_versions.txt


echo "Checking for unnecessary noqa's..."
egrep "^.{,76}[^\"]{3}# noqa: E501" test/*.py src/udar/**/*.py


echo "Running flake8..."
flake8 *.py test/**/*.py src/udar/**/*.py


echo "Running mypy..."
mypy src/udar


echo "Running pytest..."
python3.7 -m pytest --cov=udar --cov-append --cov-report term-missing --doctest-modules


rm .coverage  # can conflict with tox
echo "If everything passes, run tox."
