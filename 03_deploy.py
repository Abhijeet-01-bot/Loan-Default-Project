# 03_deploy.py

import time

from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Environment,
    CodeConfiguration,
)
from config import (
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
    MODEL_NAME,
    ENDPOINT_NAME,
    DEPLOYMENT_NAME,
)


def get_credential():
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
        return credential
    except Exception:
        print("DefaultAzureCredential failed. Opening browser login...")
        return InteractiveBrowserCredential()


def main():
    credential = get_credential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )

    print("Getting latest registered model...")

    models = list(ml_client.models.list(name=MODEL_NAME))
    if not models:
        raise ValueError(f"No model found with name: {MODEL_NAME}")

    latest_model = sorted(models, key=lambda m: int(m.version))[-1]

    print("Using model:")
    print("Name:", latest_model.name)
    print("Version:", latest_model.version)

    print("\nCreating/updating online endpoint...")

    endpoint = ManagedOnlineEndpoint(
        name=ENDPOINT_NAME,
        description="Loan default real-time inference endpoint",
        auth_mode="key",
    )

    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print("Endpoint ready.")

    print("\nCreating environment...")

    env = Environment(
        name="loan-default-inference-env",
        description="Environment for loan default model inference",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        conda_file={
            "channels": ["conda-forge"],
            "dependencies": [
                "python=3.10",
                "pip",
                {
                    "pip": [
                        "azureml-inference-server-http",
                        "pandas==2.3.3",
                        "numpy==2.2.6",
                        "scikit-learn==1.7.2",
                        "joblib==1.5.3",
                    ]
                },
            ],
        },
    )

    env = ml_client.environments.create_or_update(env)

    print("\nCreating/updating deployment...")

    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=latest_model,
        environment=env,
        code_configuration=CodeConfiguration(
            code=".",
            scoring_script="score.py",
        ),
        instance_type="Standard_DS2_v2",
        instance_count=1,
    )

    ml_client.online_deployments.begin_create_or_update(deployment).result()

    print("Deployment ready.")

    print("\nRouting 100% traffic to deployment...")

    endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    print("Traffic updated.")
    print("Endpoint name:", ENDPOINT_NAME)
    print("Deployment name:", DEPLOYMENT_NAME)

    print("\nWaiting 30 seconds before testing...")
    time.sleep(30)


if __name__ == "__main__":
    main()