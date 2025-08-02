variable "prefix" {
  description = "Prefix used for all resource names"
  type        = string
}

variable "location" {
  description = "Azure region to deploy resources"
  type        = string
  default     = "japaneast"
}

variable "container_image" {
  description = "Container image for the application"
  type        = string
}
