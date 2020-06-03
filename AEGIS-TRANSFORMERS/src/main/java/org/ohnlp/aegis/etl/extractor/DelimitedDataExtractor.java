package org.ohnlp.aegis.etl.extractor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.ohnlp.aegis.etl.DataStruct;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.stream.Stream;

/**
 * Retrieves data in delimited format
 */
public class DelimitedDataExtractor {
    // TODO change this to use input streams instead to not require loading the whole thing into memory at once...
    /**
     * Loads a character/string delimited formatted input
     * Expected format: [DATEFORMATTED-STRING][DELIMITER]prev1[DELIMITER]prev2[DELIMITER]...prevn
     * @param dateFormat the date format string to use. For examples, refer to {@link SimpleDateFormat}
     * @param delimitedFile The delimited file to load
     * @param delimiter The column delimiter, e.g. ',' or '|' or '\t', can be a regex string
     * @param headerRow Whether a header row is present in the data and should be skipped
     * @return a list of {@link DataStruct} sorted in ascending chronological order
     * @throws IOException If error occurs during data read
     * @throws ParseException If date format is incorrect
     */
    public static Stream<DataStruct> load(String dateFormat, File delimitedFile, String delimiter, boolean headerRow) throws IOException, ParseException {
        DateFormat df = new SimpleDateFormat(dateFormat);
        List<String> input =  Files.readAllLines(delimitedFile.toPath());
        List<DataStruct> ret = new ArrayList<>(headerRow ? input.size() - 1 : input.size());
        for (String line : input) {
            if (headerRow) {
                headerRow = false;
                continue;
            }
            String[] split = line.split(delimiter);
            Date date = df.parse(split[0]);
            double[] features = Arrays.stream(Arrays.copyOfRange(split, 1, split.length)).mapToDouble(Double::parseDouble).toArray();
            ret.add(new DataStruct(date, features));
        }
        ret.sort(DataStruct::compareTo);
        return ret.stream();
    }
}
