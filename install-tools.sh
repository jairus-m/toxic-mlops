#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' 

echo -e "${BLUE} Installing developer tools on macOS...${NC}\n"

# Track installation failures
FAILED_INSTALLS=()

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
        return 0
    else
        echo -e "${RED}❌ $2${NC}"
        return 1
    fi
}

# Install uv if not already installed
if command_exists uv; then
    echo -e "${YELLOW}⚠️  uv is already installed${NC}"
    uv --version
else
    echo -e "${BLUE}📦 Installing uv...${NC}"
    if command_exists brew; then
        brew install uv
        print_status $? "uv installation via Homebrew"
    else
        echo -e "${YELLOW}⚠️  Homebrew not found. Installing uv manually...${NC}"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        if print_status $? "uv installation via script"; then
            # Try to source the shell config to make uv available
            if [[ "$SHELL" == *"zsh"* ]]; then
                source ~/.zshrc 2>/dev/null || true
            elif [[ "$SHELL" == *"bash"* ]]; then
                source ~/.bashrc 2>/dev/null || true
            fi
            # Also add to current session
            export PATH="$HOME/.local/bin:$PATH"
        fi
    fi
fi

# Install task if not already installed
if command_exists task; then
    echo -e "${YELLOW}⚠️  task is already installed${NC}"
    task --version
else
    echo -e "${BLUE}📦 Installing task...${NC}"
    if command_exists brew; then
        brew install go-task
        print_status $? "task installation via Homebrew"
    else
        echo -e "${YELLOW}⚠️  Homebrew not found. Installing task manually...${NC}"
        # Install to a directory in PATH
        sudo sh -c 'curl -sL https://taskfile.dev/install.sh | sh -s -- -d /usr/local/bin'
        print_status $? "task installation via script"
    fi
fi

# Install AWS CLI if not already installed
if command_exists aws; then
    echo -e "${YELLOW}⚠️  AWS CLI is already installed${NC}"
    aws --version
else
    echo -e "${BLUE}📦 Installing AWS CLI...${NC}"
    if command_exists brew; then
        brew install awscli
        print_status $? "AWS CLI installation via Homebrew"
    else
        echo -e "${RED}❌ Homebrew not found. Please install AWS CLI manually from: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
        FAILED_INSTALLS+=("AWS CLI")
    fi
fi

# Install Terraform if not already installed
if command_exists terraform; then
    echo -e "${YELLOW}⚠️  Terraform is already installed${NC}"
    terraform --version
else
    echo -e "${BLUE}📦 Installing Terraform...${NC}"
    if command_exists brew; then
        brew install terraform
        print_status $? "Terraform installation via Homebrew"
    else
        echo -e "${RED}❌ Homebrew not found. Please install Terraform manually from: https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli"
        FAILED_INSTALLS+=("Terraform")
    fi
fi

echo -e "\n${BLUE}🔍 Verifying installations...${NC}\n"

# Verify installations and track failures
VERIFICATION_FAILED=()

if command_exists uv; then
    echo -e "${GREEN}✅ uv is installed and working:${NC}"
    uv --version
else
    echo -e "${RED}❌ uv verification failed${NC}"
    VERIFICATION_FAILED+=("uv")
fi

if command_exists task; then
    echo -e "${GREEN}✅ task is installed and working:${NC}"
    task --version
else
    echo -e "${RED}❌ task verification failed${NC}"
    VERIFICATION_FAILED+=("task")
fi

if command_exists aws; then
    echo -e "${GREEN}✅ AWS CLI is installed and working:${NC}"
    aws --version
else
    echo -e "${RED}❌ AWS CLI verification failed${NC}"
    VERIFICATION_FAILED+=("AWS CLI")
fi

if command_exists terraform; then
    echo -e "${GREEN}✅ Terraform is installed and working:${NC}"
    terraform --version
else
    echo -e "${RED}❌ Terraform verification failed${NC}"
    VERIFICATION_FAILED+=("Terraform")
fi

# Summary
echo -e "\n${BLUE}📊 Installation Summary:${NC}"

if [ ${#VERIFICATION_FAILED[@]} -eq 0 ]; then
    echo -e "${GREEN}🎉 All tools installed successfully!${NC}"
    
    if [ -f "Taskfile.yml" ]; then
        echo -e "\n${BLUE}📋 Available tasks in this project:${NC}"
        task --list
    fi

    echo -e "\n${BLUE}💡 Next steps:${NC}"
    echo -e "  • Run 'task --list' to see available tasks"
    echo -e "  • Run 'task <task-name>' to execute a specific task/assignment"
    
    exit 0
else
    echo -e "${YELLOW}⚠️  Some tools failed to install or verify:${NC}"
    for tool in "${VERIFICATION_FAILED[@]}"; do
        echo -e "  • $tool"
    done
    
    if [ ${#FAILED_INSTALLS[@]} -gt 0 ]; then
        echo -e "\n${BLUE}📋 Manual installation required for:${NC}"
        for tool in "${FAILED_INSTALLS[@]}"; do
            echo -e "  • $tool"
        done
    fi
    
    echo -e "\n${YELLOW}Please install missing tools manually and run this script again.${NC}"
    exit 1
fi