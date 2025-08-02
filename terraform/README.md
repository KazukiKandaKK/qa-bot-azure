# Infrastructure as Code

This directory contains Terraform configuration for deploying the Azure resources
used by the QA bot. The structure separates reusable modules from environment
specific configurations.

```
terraform/
  modules/
    base/         # Module provisioning API Management, Container Apps, etc.
  dev/            # Development environment using the base module
```

To deploy the dev environment:

```sh
cd terraform/dev
terraform init
terraform apply
```
