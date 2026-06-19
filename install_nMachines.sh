#!\bin\bash

cd
mkdir -p Gitlab
cd Gitlab
git clone git@gitlab.epfl.ch:risharma/decentralizepy.git
cd decentralizepy
mkdir -p leaf/data/femnist/data/train
mkdir -p leaf/data/femnist/data/test
mkdir -p leaf/data/femnist/per_user_data/train
~/miniforge3/bin/conda remove --name decpy --all
~/miniforge3/bin/conda create -n decpy python=3.9
~/miniforge3/envs/decpy/bin/pip install --upgrade pip --quiet
~/miniforge3/envs/decpy/bin/pip install --editable .\[dev\]
