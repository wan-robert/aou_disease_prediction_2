# --- AUTO-GENERATED ENVIRONMENT SETUP ---

Sys.setenv(WORKSPACE_CDR = "wb-silky-artichoke-2408.C2024Q3R8")
Sys.setenv(GOOGLE_CLOUD_PROJECT = "wb-sunny-radish-6214")
Sys.setenv(WORKSPACE_BUCKET = "gs://workspace-bucket-wb-sunny-radish-6214")
Sys.setenv(WORKSPACE_TEMP_BUCKET = "gs://temporary-workspace-bucket-wb-sunny-radish-6214")
Sys.setenv(EXPORT_BUCKET = "gs://workspace-bucket-wb-sunny-radish-6214")

# --- VERIFICATION BLOCK ---

vars_to_check <- c("WORKSPACE_CDR", "GOOGLE_CLOUD_PROJECT", "WORKSPACE_BUCKET", "WORKSPACE_TEMP_BUCKET", "EXPORT_BUCKET")

cat("\n🔍 Current Workspace Variables:\n")
for (v in vars_to_check) {
  value <- Sys.getenv(v)
  # Prints the name (padded to 22 chars) and the value
  cat(sprintf("  %-22s : %s\n", v, value))
}
message("\n✅ Environment Loaded Successfully.")
