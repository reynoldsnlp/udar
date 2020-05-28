
echo -n "Upgrade HFST and vislcg3 (y/n)? "
read answer

if [ "$answer" != "${answer#[Yy]}" ] ; then
    curl https://apertium.projectjj.com/osx/install-nightly.sh | sudo bash
else
    echo "Not upgrading hfst and vislcg3."
fi

hfst-tokenize --version | grep hfst > hfst_vislcg3_versions.txt
vislcg3 --version | grep VISL >> hfst_vislcg3_versions.txt

flake8 *.py test/**/*.py udar/**/*.py
mypy udar
pytest --cov=udar --cov-append --cov-report term-missing --doctest-modules
rm .coverage  # can conflict with tox

echo "If everything passes, run tox."
