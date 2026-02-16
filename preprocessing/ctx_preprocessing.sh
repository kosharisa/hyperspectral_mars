# TODO: clean folder paths before commit

CTX_FOLDER="PATH_TO_CTX_FILES_TO_BE_PROCESSED"
CTX_PROCESSED_FOLDER="OUTPUT_PATH_FOR_PROCESSED"
MAP_TEMP_FOLDER="PATH_TO_MAP_PROJECTIONS_FILES" # this directory will be filled with necessary files during ISIS installation and should be something like $ISIS_ENVIRONMENT$/appdata/templates/maps

for file in "$CTX_FOLDER"/*img; do
    if [ -e "$file" ]; then
        echo "Processing $file..."

        base_name=$(basename "$file" .img)

        echo "Running mroctx2isis..."
        mroctx2isis from="$file" to="${CTX_PROCESSED_FOLDER}/${base_name}.cub"
        
        echo "Running spiceinit..."
        spiceinit from="${CTX_PROCESSED_FOLDER}/${base_name}.cub"
        
        echo "Running ctxcal..."
        ctxcal from="${CTX_PROCESSED_FOLDER}/${base_name}.cub" to="${CTX_PROCESSED_FOLDER}/${base_name}.cal.cub"
        
        echo "Running cam2map..."
        cam2map from="${CTX_PROCESSED_FOLDER}/${base_name}.cal.cub" to="${CTX_PROCESSED_FOLDER}/${base_name}.cal.eq.cub" map="${MAP_TEMP_FOLDER}/equirectangular.map"
        
        echo "Finished processing $file"
    else
        echo "No .img files found in $CTX_FOLDER"
        break
        
    fi
    
done