
# Manually run below to setup the python environment

# A) Setup instructions for new machine
#git clone https://github.com/geco-bern/swiss-ndvi-processing.git
#cd swiss-ndvi-processing
#sudo apt install python3-virtualenv # instead of pip install virtualenv
#virtualenv .env/ndvi
#source .env/ndvi/bin/activate
#pip install -r requirements.txt     # Note that this can easily take 1h depending on the internet connection.


# B) Only needed for development to setup git and GitHub user:
# ssh-keygen -t ed25519 -C "fabian-bernhard@swisstopo-tunder" # define a passphrase
# eval "$(ssh-agent -s)"   # Do this again later if connection issues come up
# ssh-add ~/.ssh/id_ed25519
# Manually add the key to Github: 
# # go to https://github.com/settings/keys
# # New SSH key, give title: fabian-bernhard@swisstopo-tunder
# # Leave as "Authentication key"
# # and copy paste content of public key, i.e. `less ~/.ssh/id_ed25519.pub`
#
# Don't forget to delete this from your GitHub account when not needed anymore.
#
# # eventually test with: `ssh -T git@github.com`
#
# # Fix Git remotes, just in case you had set it up with HTTPS instead of SSH:
# git remote set-url origin git@github.com:geco-bern/swiss-ndvi-processing
# git remote -v # to check
#
# # Set up git user name and email
# # git config --global user.name "fabern"
# # git config --global user.email "10245680+fabern@users.noreply.github.com"   # get it from: https://github.com/settings/emails

# C) Check system specifications agree with below:
# (ndvi) fabian-bernhard@tunder:/home/Shared/UniBe-swiss-ndvi/data$ uname -or
# 6.8.0-90-generic GNU/Linux
# (ndvi) fabian-bernhard@tunder:/home/Shared/UniBe-swiss-ndvi/data$ lsb_release -irc
# No LSB modules are available.
# Distributor ID: Ubuntu
# Release:        24.04
# Codename:       noble

# D)