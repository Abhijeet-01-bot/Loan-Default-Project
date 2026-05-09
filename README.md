## Environment Variables Required

Create a `.env` file in the root directory and add the following variables:

```env
# Azure Subscription Details
SUBSCRIPTION_ID=
RESOURCE_GROUP=
WORKSPACE_NAME=
LOCATION=

# Azure ML Compute
COMPUTE_NAME=
EXPERIMENT_NAME=

# Dataset & Model
DATA_FILE=
MODEL_NAME=

# Deployment Details
ENDPOINT_NAME=
DEPLOYMENT_NAME=

# Deployment VM Configuration
VM_SIZE=
```

---

## Project Setup

### 1. Create Virtual Environment

```bash
python -m venv myenv
```

---

### 2. Activate Virtual Environment

#### Linux / macOS
```bash
source myenv/bin/activate
```

#### Windows
```powershell
myenv\Scripts\activate
```

---

## Run Scripts Sequentially

Execute the following scripts one by one in order:

### Step 1 — Setup Azure Resources
```bash
python 01_setup_azure.py
```

This script:
- Connects to Azure
- Creates or connects to Azure ML Workspace
- Sets up compute resources

---

### Step 2 — Train & Register Model
```bash
python 02_train_register.py
```

This script:
- Loads dataset
- Trains the ML model
- Registers the model into Azure ML

---

### Step 3 — Deploy Model Endpoint
```bash
python 03_deploy.py
```

This script:
- Creates online endpoint
- Deploys the registered model
- Exposes inference API

---

## Notes

- Ensure Azure CLI is installed and authenticated:
  
```bash
az login
```

- Make sure all environment variables are correctly configured before running the scripts.

- Run the scripts in the exact order mentioned above.