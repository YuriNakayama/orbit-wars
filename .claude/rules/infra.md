---
paths:
  - "infra/**"
---

# Infrastructure Rules

## General Principles

- Follow HashiCorp Configuration Language (HCL) conventions
- Write in a declarative style
- Make resource dependencies clear
- Separate configurations by environment
- Increase reusability through modularization

## Scope

Orbit Wars submission artifacts run on Kaggle, so `infra/` is used **only when training infrastructure (GPU / large-scale self-play clusters etc.) is required**. Even though it is currently unused, the shared rules below are kept here for future cloud-based training environments.

## Module Structure

A layered module structure is recommended:

```
infra/
  environments/          Environment configs (dev, staging, prod, state)
    dev/
      main.tf
      variables.tf
      outputs.tf
      terraform.tfvars   (gitignored)
    staging/
    prod/
  modules/
    foundation/          Core infrastructure (networking, IAM)
    platform/            Platform services (compute, storage, monitoring)
    applications/        Application-specific stacks (training jobs, replay store)
```

### Module Design Principles

- Clearly separate responsibilities (foundation, platform, applications)
- Do not define `provider` blocks in shared modules
- Define `required_providers` in root modules
- Manage inter-module dependencies using `outputs`

## Resource Naming

- Use `snake_case`
- Include environment or project name as prefix
- Use clear names without abbreviations
- Do not repeat resource type in resource names

```hcl
# GOOD
resource "aws_instance" "web_server" { }

# BAD
resource "aws_instance" "aws_instance_web_server" { }
```

## Variable Definitions

- Define variables in `variables.tf`
- Always specify `description` and `type`
- Set `default` values when appropriate
- Only parameterize environment-specific values

```hcl
variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "instance_count" {
  description = "Number of instances to create"
  type        = number
  default     = 1
}
```

## Conditionals and Loops

- Use `count` for conditional resource creation
- Use `for_each` for repeating resource creation
- Pre-calculate complex conditions in `locals`

```hcl
locals {
  is_production = var.environment == "prod"
}

resource "aws_instance" "example" {
  count         = var.create_instance ? 1 : 0
  instance_type = local.is_production ? "t3.large" : "t3.micro"
}

resource "aws_subnet" "private" {
  for_each          = var.availability_zones
  availability_zone = each.value
}
```

## Tag Management

- Apply common tags to all resources via `default_tags`
- Set resource-specific tags appropriately
- Use PascalCase for tag keys

```hcl
provider "aws" {
  default_tags {
    tags = {
      Environment = var.environment
      Project     = "OrbitWars"
      ManagedBy   = "Terraform"
    }
  }
}

resource "aws_instance" "web" {
  tags = {
    Name = "web-server"
    Role = "Frontend"
  }
}
```

## Secrets Management

- Do not commit `.tfvars` files to Git
- Use `TF_VAR_*` environment variables for sensitive information
- Set `sensitive = true` for outputs containing sensitive information
- Leverage AWS Secrets Manager

```hcl
variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

output "db_connection_string" {
  value     = aws_db_instance.main.endpoint
  sensitive = true
}
```

## State Management

- Use remote state (S3 + DynamoDB)
- Enable state locking to prevent concurrent modifications
- Encrypt state files
- Never commit state files to Git

```hcl
terraform {
  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "orbit-wars/terraform.tfstate"
    region         = "ap-northeast-1"
    encrypt        = true
    dynamodb_table = "terraform-lock"
  }
}
```

## Data Sources

- Use data sources for existing resources
- Prefer data sources over hardcoded values
- Cache data source results in `locals` when used multiple times

```hcl
data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
}

resource "aws_instance" "web" {
  ami = data.aws_ami.amazon_linux.id
}
```

## Output Values

- Document all outputs with descriptions
- Output values needed by other modules or external tools
- Mark sensitive outputs appropriately

## Validation

Before applying:

```bash
terraform fmt        # Consistent formatting
terraform validate   # Configuration validity
terraform plan       # Preview changes — review before apply
```

## Security Checklist

- [ ] No secrets in code or `.tfvars` committed to Git
- [ ] IAM policies follow least privilege
- [ ] Security groups are restrictive (no `0.0.0.0/0` on sensitive ports)
- [ ] Database not publicly accessible
- [ ] Encryption enabled at rest and in transit
- [ ] State files encrypted with S3 server-side encryption
- [ ] VPC flow logs enabled
- [ ] WAF configured for public-facing endpoints
