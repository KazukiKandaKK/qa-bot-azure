terraform {
  required_version = ">= 1.0"
}

module "base" {
  source          = "../modules/base"
  prefix          = var.prefix
  location        = var.location
  container_image = var.container_image
}

output "resource_group" {
  value = module.base.resource_group_name
}

output "app_url" {
  value = module.base.container_app_url
}
