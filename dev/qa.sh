#!/bin/bash

rm -f .coverage  # can conflict with tox

echo -n "Upgrade HFST and vislcg3? (y/n) "
read answer
if [ "$answer" != "${answer#[Yy]}" ] ; then
    echo "Type password: "
    curl https://apertium.projectjj.com/osx/install-nightly.sh | sudo bash
else
    echo "Not upgrading hfst and vislcg3."
fi

echo "Versions with which tests passed for this commit:" \
    > hfst_vislcg3_versions.txt
hfst-tokenize --version | grep hfst >> hfst_vislcg3_versions.txt
vislcg3 --version | grep VISL >> hfst_vislcg3_versions.txt

flake8 *.py test/**/*.py udar/**/*.py
mypy udar
pytest --cov=udar --cov-append --cov-report term-missing --doctest-modules

rm .coverage  # can conflict with tox
echo "If everything passes, run tox."
