variable "prefix" {
  description = "Prefix used for all resource names"
  type        = string
  default     = "qa-bot"
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "japaneast"
}

variable "container_image" {
  description = "Container image reference"
  type        = string
  default     = "mcr.microsoft.com/azuredocs/containerapps-helloworld:latest"
}
