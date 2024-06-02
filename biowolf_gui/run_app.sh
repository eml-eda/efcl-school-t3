if test -d ./lib/
then
    COLOR='\033[0;32m' #green
    echo -e "${COLOR}Lib exists, skipping lib download${NC}\n"
    # then echo "lib exist, skipping lib download"
else
    COLOR='\033[0;31m' #Red
    NC='\033[0m' # No Color
    echo -e "${COLOR}Lib not detected, downloading...${NC}\n"
    # echo "Crearing directory..."
    COLOR='\033[0;32m' #green
    echo -e "${COLOR}Crearing directory...${NC}\n"
    mkdir lib
    COLOR='\033[0;32m' #green
    echo -e "${COLOR}Directory created..${NC}\n"
    # echo "Directory created.."
    cd lib
    COLOR='\033[0;32m' #green
    echo -e "${COLOR}Donwloading OpenFX${NC}\n"
    # echo "Donwloading OpenFX"
    wget https://download2.gluonhq.com/openjfx/14.0.1/openjfx-14.0.1_linux-x64_bin-sdk.zip
    COLOR='\033[0;32m' #green
    echo -e "${COLOR}Installing OpenFX${NC}\n"
    # echo "Installing OpenFX"
    unzip openjfx-14.0.1_linux-x64_bin-sdk.zip
    COLOR='\033[0;32m' #green
    echo -e "${COLOR}Downloading JDK${NC}\n"
    # echo "Downloading JDK"
    wget https://download.java.net/java/GA/jdk14.0.1/664493ef4a6946b186ff29eb326336a2/7/GPL/openjdk-14.0.1_linux-x64_bin.tar.gz
    
    COLOR='\033[0;32m' #green
    echo -e "${COLOR}Installing JDK${NC}\n"
    # echo "Installing JDK"

    tar -xvzf openjdk-14.0.1_linux-x64_bin.tar.gz
    COLOR='\033[0;32m' #green
    echo -e "${COLOR}Cleaning...${NC}\n"
    # echo "Cleaning..."
    rm openjdk-14.0.1_linux-x64_bin.tar.gz
    rm openjfx-14.0.1_linux-x64_bin-sdk.zip
    cd ..
    COLOR='\033[0;32m' #green
    echo -e "${COLOR}Lib created...${NC}\n"
    # echo "Lib created..."
fi

python serialoverwrite.py &
chmod +x lib/jdk-14.0.1/bin/java
lib/jdk-14.0.1/bin/java  --module-path lib/javafx-sdk-14.0.1/lib --add-modules javafx.controls,javafx.fxml --add-exports=javafx.graphics/com.sun.javafx.css=ALL-UNNAMED -Dfile.encoding=windows-1252 -jar application/v0424/BIOWOLF.jar
