# Load required datasets
autoswithout <- read.csv("autoswithout.csv")
model_and_brand <- read.csv("model_and_brand.csv")

# Step 1: Standardize and preprocess the name field
# Convert all characters in the 'name' column to lowercase for consistent matching
autoswithout$name=tolower(autoswithout$name)
# Split the 'name' field by underscores to separate out components for easier processing
autoswithout$split=strsplit(autoswithout$name,"_")
# Initialize 'brand' and 'model' columns as NA
autoswithout$brand <- NA
autoswithout$model <- NA


# Step 2: Clean and standardize brand names in split components
# Handle special cases where brand names need standardization
autoswithout$split = lapply(autoswithout$split, function(x) {
  x <- sub("benz", "", x)                    # Remove standalone 'benz'
  x <- sub("mercedes", "mercedes_benz", x)   # Standardize Mercedes-Benz naming
  x <- sub("alfa", "alfa_romeo", x)          # Standardize Alfa_Romeo naming
  x <- sub("freelander", "land_rover", x)    # Standardize 'freelander' to 'land_rover'
  x <- gsub("[?!./*]", "", x)                # Remove special characters to avoid matching issues
  return(x)
})


# Step 3: First pass - Direct brand matching
# Loop through each entry to identify the car brand by unique matches in 'split'
for (i in 1:nrow(autoswithout)) {
  for (m in 1:nrow(model_and_brand)) {
    brand <- model_and_brand$brand[m] # If brand name is found in split components, assign it
    # Check if the brand uniquely matches a term in 'split'
    if ((brand %in% autoswithout$split[[i]])) { 
      autoswithout$brand[i] <- brand
      break
    } else {
      autoswithout$brand[i] <- NA
    }
  }
}


# Step 4: Second pass - Model-based brand inference
# If brand wasn't found directly, try to infer it from model information
for (i in 1:nrow(autoswithout)) {
  for (m in 1:nrow(model_and_brand)) {
    model <- model_and_brand$model[m]
    # If model is found in split components, get its corresponding brand
    if ((model %in% autoswithout$split[[i]])) { 
      autoswithout$model[i] <- model 
      break
    } else {
      autoswithout$model[i] <- NA 
    }
  }
}
#Step 4.1
#Handle situations where multiple models or brands could match; assign NA if ambiguous
model_all<-model_and_brand$model 
brand_all<-model_and_brand$brand 
for(i in 1:length(autoswithout$split)){
  a<-autoswithout$split[[i]]
  b<-model_all
  c<-brand_all
  judge1<- a %in% b # Check if any part matches a model
  judge2<- a %in% c # Check if any part matches a brand

  if(sum(judge1)>1){ 
    autoswithout$model[i]<- NA # Multiple model matches found - mark as NA
  }
  else if(sum(judge1)==0){ # If no model match, try to infer from brand
    if(sum(judge2)==1){ # Single brand match found
      index <- which(c == a[judge2])
      if (length(index)==1){
        autoswithout$model[i]<- b[index]
      }# Assign model if brand has unique model
      else { # Multiple models for brand - mark as NA
        autoswithout$model[i]<- NA
      }
    }
    else if(sum(judge2)>1){ # Multiple brand matches - mark as NA
      autoswithout$model[i]<- NA
    }
    else{ # No matches found - mark as NA
      autoswithout$model[i]<- NA # If multiple brands match, assign NA to brand
    }
  }
#Step 4.2
# Handle brand assignment logic (similar to model logic above)
  if(sum(judge2)>1){ 
    autoswithout$brand[i]<- NA
  }
  else if(sum(judge2)==0){ 
    if(sum(judge1)==1){
      index <- which(b == a[judge1])
      if (length(index)==1){
        autoswithout$brand[i]<- c[index]
      }
      else { 
        autoswithout$brand[i]<- NA
      }
    }
    else if(sum(judge1)>1){ 
      autoswithout$brand[i]<- NA
    }
    else{ 
      autoswithout$brand[i]<- NA
    }
  }
}


# Step 5: Handle specific cases for BMW, Ford, Mazda, Mercedes-Benz, Honda and Volkswagen
# Set brand and model based on patterns specific to each brand's model line-up

#Step 5.1 
# Special handling for BMW models - handle numeric series identifiers
for(i in 1:length(autoswithout$split)){
  x<-autoswithout$split[[i]]
  judge<-"bmw" %in% x
  if(sum(judge)>=1){
    for(j in 1:length(x)){
      result1 <- grepl("^3", x[j])
      result2 <- grepl("^5", x[j])
      result3 <- grepl("^1", x[j])
      result4 <- grepl("^x", x[j])
      result5 <- grepl("^7", x[j])
      result6 <- grepl("^z", x[j])
      result7 <- grepl("^m", x[j])
      result8 <- grepl("^6", x[j])
      # Assign appropriate model based on series identifier
      if(result1==T){autoswithout$model[i]<- "3er" 
      break  }
      else if(result2==T){autoswithout$model[i]<- "5er"
        break}
      else if(result3==T){autoswithout$model[i]<- "1er"
        break}
      else if(result4==T){autoswithout$model[i]<- "x_reihe"
        break}
      else if(result5==T){autoswithout$model[i]<- "7er"
        break}
      else if(result6==T){autoswithout$model[i]<- "z_reihe"
        break}
      else if(result7==T){autoswithout$model[i]<- "m_reihe"
        break}
      else if(result8==T){autoswithout$model[i]<- "6er"
        break}
    }
  }
}
#Step 5.2
# Special handling for Ford models
for(i in 1:length(autoswithout$split)){
  x<-autoswithout$split[[i]]
  judge<-"ford" %in% x
  if(sum(judge)>=1){
    for(j in 1:length(x)){
      result1 <- grepl("^c", x[j]) # Match C-Max
      result2 <- grepl("^s", x[j]) # Match S-Max
      if(result1==T){
        autoswithout$model[i]<- "c_max"
        break
      }
      else if(result2==T){
        autoswithout$model[i]<- "s_max"
        break
      }
    }
  }
}
#Step 5.3
# Special handling for Mazda models
for(i in 1:length(autoswithout$split)){
  x<-autoswithout$split[[i]]
  judge<-"mazda" %in% x
  if(sum(judge)>=1){
    for(j in 1:length(x)){
      result1 <- grepl("^6", x[j])
      result2 <- grepl("^mx", x[j])
      result3 <- grepl("^3", x[j])
      result4 <- grepl("^cx", x[j])
      result5 <- grepl("^rx", x[j])
      result6 <- grepl("^1", x[j])
      result7 <- grepl("^5", x[j])
      if(result1==T){
        autoswithout$model[i]<- "6_reihe"
        break
      }
      else if(result2==T){
        autoswithout$model[i]<- "mx_reihe"
        break
      }
      else if(result3==T){
        autoswithout$model[i]<- "3_reihe"
        break
      }
      else if(result4==T){
        autoswithout$model[i]<- "cx_reihe"
        break
      }
      else if(result5==T){
        autoswithout$model[i]<- "rx_reihe"
        break
      }
      else if(result6==T){
        autoswithout$model[i]<- "1_reihe"
        break
      }
      else if(result7==T){
        autoswithout$model[i]<- "5_reihe"
        break
      }
    }
  }
}
#Step 5.4
# Special handling for Mercedes-Benz models - handle class identifiers
for(i in 1:length(autoswithout$split)){
  x<-autoswithout$split[[i]]
  judge<-"mercedes_benz" %in% x
  l<-length(x)
  x<-x[2:l] # Skip first element as it's the brand name
  if(sum(judge)>=1){
    for(j in 1:length(x)){
      result1 <- grepl("^c", x[j])
      result2 <- grepl("^e", x[j])
      result3 <- grepl("^a", x[j])
      result4 <- grepl("^m", x[j])
      result5 <- grepl("^s", x[j])
      result6 <- grepl("^b", x[j])
      result7 <- grepl("^v", x[j])
      result8 <- grepl("^g", x[j])
      # Assign appropriate model based on class identifier
      if(result1==T){
        autoswithout$model[i]<- "c_klasse"
        break
      }
      else if(result2==T){
        autoswithout$model[i]<- "e_klasse"
        break
      }
      else if(result3==T){
        autoswithout$model[i]<- "a_klasse"
        break
      }
      else if(result4==T){
        autoswithout$model[i]<- "m_klasse"
        break
      }
      else if(result5==T){
        autoswithout$model[i]<- "s_klasse"
        break
      }
      else if(result6==T){
        autoswithout$model[i]<- "b_klasse"
        break
      }
      else if(result7==T){
        autoswithout$model[i]<- "v_klasse"
        break
      }
      else if(result8==T){
        autoswithout$model[i]<- "g_klasse"
        break
      }
    }
  }
}
#Step 5.5
# Special handling for Honda models
for(i in 1:length(autoswithout$split)){
  x<-autoswithout$split[[i]]
  judge<-"honda" %in% x
  if(sum(judge)>=1){
    for(j in 1:length(x)){
      result1 <- grepl("^cr", x[j]) # Match CR series
      if(result1==T){
        autoswithout$model[i]<- "cr_reihe"
        break
      }
    }
  }
}

# Check number of unmatched entries
sum(is.na(autoswithout$model))
sum(is.na(autoswithout$brand))
sum(is.na(autoswithout$model) & is.na(autoswithout$brand))


# Step 6: Check for pattern-based matches in remaining NA entries
# Create subset of unmatched entries for further analysis
Return_NA<-autoswithout[is.na(autoswithout$model) & is.na(autoswithout$brand), ]
for (i in 1:length(Return_NA$split)) {
  x <- Return_NA$split[[i]][1]
  id <- Return_NA$X[[i]]
# Step 6.1:
# Match Mercedes-Benz classes and BMW series based on first character/number
  if (x == "c") {
    autoswithout$model[which(autoswithout$X == id)] <- "c_klasse"
    autoswithout$brand[which(autoswithout$X == id)] <- "mercedes_benz"
  } else if (x == "e") {
    autoswithout$model[which(autoswithout$X == id)] <- "e_klasse"
    autoswithout$brand[which(autoswithout$X == id)] <- "mercedes_benz"
  } else if (x == "a") {
    autoswithout$model[which(autoswithout$X == id)] <- "a_klasse"
    autoswithout$brand[which(autoswithout$X == id)] <- "mercedes_benz"
  } else if (x == "m") {
    autoswithout$model[which(autoswithout$X == id)] <- "m_klasse"
    autoswithout$brand[which(autoswithout$X == id)] <- "mercedes_benz"
  } else if (x == "s") {
    autoswithout$model[which(autoswithout$X == id)] <- "s_klasse"
    autoswithout$brand[which(autoswithout$X == id)] <- "mercedes_benz"
  } else if (x == "b") {
    autoswithout$model[which(autoswithout$X == id)] <- "b_klasse"
    autoswithout$brand[which(autoswithout$X == id)] <- "mercedes_benz"
  } else if (x == "v") {
    autoswithout$model[which(autoswithout$X == id)] <- "v_klasse"
    autoswithout$brand[which(autoswithout$X == id)] <- "mercedes_benz"
  } else if (x == "g") {
    autoswithout$model[which(autoswithout$X == id)] <- "g_klasse"
    autoswithout$brand[which(autoswithout$X == id)] <- "mercedes_benz"
  } else if (grepl("^3", x)) {
    autoswithout$model[which(autoswithout$X == id)] <- "3er"
    autoswithout$brand[which(autoswithout$X == id)] <- "bmw"
  } else if (grepl("^5", x)) {
    autoswithout$model[which(autoswithout$X == id)] <- "5er"
    autoswithout$brand[which(autoswithout$X == id)] <- "bmw"
  } else if (grepl("^1", x)) {
    autoswithout$model[which(autoswithout$X == id)] <- "1er"
    autoswithout$brand[which(autoswithout$X == id)] <- "bmw"
  } else if (grepl("^7", x)) {
    autoswithout$model[which(autoswithout$X == id)] <- "7er"
    autoswithout$brand[which(autoswithout$X == id)] <- "bmw"
  } else if (grepl("^6", x)) {
    autoswithout$model[which(autoswithout$X == id)] <- "6er"
    autoswithout$brand[which(autoswithout$X == id)] <- "bmw"
  }
}
# Step 6.2:
# Handle Volkswagen VW abbreviation cases
for(i in 1:length(autoswithout$split)){
  x<-autoswithout$split[[i]]
  judge<-"vw" %in% x
  if(sum(judge)>=1){
    autoswithout$brand[i]<- "volkswagen"
  }
}
# Handle cases where name starts with 'vw'
for(i in 1:length(autoswithout$split)){
  x<-autoswithout$split[[i]]
  judge<-grepl("^vw",x)
  if(sum(judge)>=1){
    autoswithout$brand[i]<- "volkswagen"
  }
}
# Step 6.3:
# Handle cases where name starts with 'bmw'
for(i in 1:length(autoswithout$split)){
  x<-autoswithout$split[[i]]
  judge<-grepl("^bmw",x)
  if(sum(judge)>=1){
    autoswithout$brand[i]<- "bmw"
  }
}
# Step 6.4:
# Handle Golf models (always Volkswagen)
for(i in 1:length(autoswithout$split)){
  x<-autoswithout$split[[i]]
  judge<-grepl("^golf",x)
  if(sum(judge)>=1){
    autoswithout$model[i]<- "golf"
    autoswithout$brand[i]<- "volkswagen"
  }
}

# Step 6.5:
# Handle mazda
for(i in 1:length(autoswithout$split)){
  x<-autoswithout$split[[i]]
  judge<-grepl("^mazda",x)
  if(sum(judge)>=1){
    autoswithout$brand[i]<- "mazda"
  }
}
#Step 7: Final attempt to match brands in remaining NA cases
for (i in 1:nrow(Return_NA)) {
  for (m in 1:nrow(model_and_brand)) {
    brand <- model_and_brand$brand[m] 
    if ((brand %in% Return_NA$split[[i]])) { 
      id <- Return_NA$X[[i]]
      autoswithout$brand[which(autoswithout$X == id)] <- brand
      break}}}

#Step 8: Output the final NA results to check the remaining rows without brand/model information
# Summarize the number of unmatched brands and models
sum(is.na(autoswithout$model))
sum(is.na(autoswithout$brand))
sum(is.na(autoswithout$model) & is.na(autoswithout$brand))
Return_NA<-autoswithout[is.na(autoswithout$model) & is.na(autoswithout$brand), ]
print(head(Return_NA))
