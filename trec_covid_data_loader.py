import json
import os
import io
import pandas as pd
import pickle


class TrecCovidDatasetManager:

    def __init__(self, data_folder_path, metadata_file_path):
        self.metadata_file_path = metadata_file_path
        self.data_folder_path = data_folder_path
        self.metadata_dict = {}
        self.paper_dict = {}

    # Reads the metadata file and store its information in a dict
    def load_metadata_from_csv(self):
        """
        Load the registries from the metadata file in csv format to a dictionary. This method should be executed before
        trying to retrieve any document from the collection
        :return: None
        """

        df = pd.read_csv(self.data_folder_path + 'metadata.csv', low_memory=False, dtype=str)

        # for index, cord_uid, sha, source_x, title, doi, pmcid, pubmed_id, license, abstract, publish_time, \
        #     authors, journal, mag_id, who_covidence_id, arxiv_id, pdf_json_files, pmc_json_files, url, s2_id \
        #         in df.itertuples():

        for index, cord_uid, sha, source_x, title, doi, pmcid, pubmed_id, license, abstract, publish_time, \
            authors, journal, mag_id, who_covidence_id, arxiv_id, pdf_json_files, pmc_json_files, url \
                in df.itertuples():

            # Check there are no repeated keys
            if cord_uid not in self.metadata_dict:
                # Check there is a pdf or pmc file corresponding to that key
                if not (pd.isna(pdf_json_files) and pd.isna(pmc_json_files)):
                    self.metadata_dict[cord_uid] = {'title': title,
                                                    'abstract': abstract,
                                                    'pdf_file': pdf_json_files,
                                                    'pmc_file': pmc_json_files}

    # Saves metadata dict to disk
    def save_metadata_as_pickle(self):
        """
        Saves the metadata dictionary into a metadata.pickle file
        :return: None
        """

        with open(self.data_folder_path + 'metadata.pickle', 'wb') as file_handle:
            pickle.dump(self.metadata_dict, file_handle, pickle.HIGHEST_PROTOCOL)

    # Loads the previously saved metadata dict
    def load_metadata_from_pickle(self):
        """
        Loads the metadata dictionary from the previously saved metadata.pickle file
        :return:
        """

        with open(self.data_folder_path + 'metadata.pickle', 'rb') as file_handle:
            self.metadata_dict = pickle.load(file_handle)

    # Loads a document data from a json file
    def _load_doc_from_json_(self, doc_file_path):
        """
        Loads a document from its json file
        :param doc_file_path: the file path of the json document
        :return: a dictionary containing the document attributes including title and text
        """
        doc_json_file = io.open(file=self.data_folder_path + doc_file_path, mode='r', encoding='utf-8')
        doc_json = json.load(doc_json_file)

        paper_title = doc_json['metadata']['title']

        paper_body_text = []

        for t in doc_json['body_text']:
            paper_body_text.append(t['text'])

        doc_json_file.close()

        return {'title': paper_title, 'text': paper_body_text}

    # Given a cord_uid returns a document in dict format from its json file. The text att is a list of str
    def get_document_from_jsom(self, cord_uid):
        """
        Given a cord_uid returns the document with the matching id. This method reads the document from the json file.
        :param cord_uid: The id of the document to retrieve
        :return: a dictionary containing the document attributes including cord_uid, title and text
        """

        if cord_uid not in self.metadata_dict:
            raise Exception("Provided cord_uid does not match any document in our dataset")

        metadata = self.metadata_dict[cord_uid]

        doc = {}
        if not pd.isna(metadata['pmc_file']):
            # Check if the file exists in disk
            pmc_file = metadata['pmc_file']
            if not os.path.isfile(self.data_folder_path + pmc_file):
                raise Exception("Provided cord_uid does not match any document in our dataset")

            doc = self._load_doc_from_json_(pmc_file)
        else:
            # Check if the file exists in disk
            pdf_file = metadata['pdf_file']
            if not os.path.isfile(self.data_folder_path + pdf_file):
                raise Exception("Provided cord_uid does not match any document in our dataset")

            doc = self._load_doc_from_json_(pdf_file)

        doc['cord_uid'] = cord_uid

        return doc

    # Given a cord_uid returns a document in dict format from its json file. The text att is a single str
    def get_document_from_jsom_no_paragraph_list(self, cord_uid):
        """
        Given a cord_uid returns the document with the matching id. This method reads the document from the json file.
        The 'text' attribute of the document is a single string instead of a list of paragraphs
        :param cord_uid: The id of the document to retrieve
        :return: a dictionary containing the document attributes including cord_uid, title and text
        """

        if cord_uid not in self.metadata_dict:
            raise Exception("Provided cord_uid does not match any document in our dataset")

        metadata = self.metadata_dict[cord_uid]

        pre_doc = {}
        if not pd.isna(metadata['pmc_file']):
            # Check if the file exists in disk
            pmc_file = metadata['pmc_file']
            if not os.path.isfile(self.data_folder_path + pmc_file):
                raise Exception("Provided cord_uid does not match any document in our dataset")

            # load the file data
            pre_doc = self._load_doc_from_json_(metadata[pmc_file])
        else:
            # Check if the file exists in disk
            pdf_file = metadata['pdf_file']
            if not os.path.isfile(self.data_folder_path + pdf_file):
                raise Exception("Provided cord_uid does not match any document in our dataset")

            # load the file data
            pre_doc = self._load_doc_from_json_(pdf_file)

        doc = {'cord_uid': cord_uid, 'title': pre_doc['title'], 'text': ''.join(pre_doc['text'])}

        return doc

    # Creates an in-memory dict with all the documents (can be memory-expensive)
    def create_papers_dict(self):
        """
        Reads all the documents and store them in a dictionary using the cord_uid's as keys. This method can be
        memory-expensive
        :return: None
        """

        for cord_uid, metadata in self.metadata_dict.items():
            doc = {}
            pmc_file = metadata['pmc_file']
            pdf_file = metadata['pdf_file']
            if not pd.isna(pmc_file):
                if os.path.isfile(self.data_folder_path + pmc_file):
                    doc = self._load_doc_from_json_(pmc_file)
            else:
                if os.path.isfile(self.data_folder_path + pdf_file):
                    doc = self._load_doc_from_json_(pdf_file)

            # Check if a document was found in the folders
            if doc:
                doc['cord_uid'] = cord_uid
                self.paper_dict[cord_uid] = doc

    # Saves to disc the in-memory dict of documents
    def save_docs_dict_as_pickle(self):
        """
        Saves the documents dictionary to a docs_dict.pickle file.
        :return: None
        """

        with open(self.data_folder_path + 'docs_dict.pickle', 'wb') as file_handle:
            pickle.dump(self.paper_dict, file_handle, pickle.HIGHEST_PROTOCOL)

    # Loads the previously saved dict of documents
    def load_docs_dict_from_pickle(self):
        """
        Loads the previously created docs_dict.pickle file containing the ditionary with all the documents in the
        dataset
        :return: None
        """

        with open(self.data_folder_path + 'docs_dict.pickle', 'rb') as file_handle:
            self.paper_dict = pickle.load(file_handle)

    # Given a cord_uid returns a document in dict format from the in-memory dict. The text att is a list of str
    def get_document_from_dict(self, cord_uid):
        """
        Given a cord_uid returns the document with the matching id. This method reads the document from the internal
        documents dictionary where all documents have to be previously stored.
        :param cord_uid: The id of the document to retrieve
        :return: a dictionary containing the document attributes including cord_uid, title and text
        """

        if cord_uid not in self.paper_dict:
            raise Exception("Provided cord_uid does not match any document in our dataset")

        doc = self.paper_dict[cord_uid]
        return doc

    # Given a cord_uid returns a document in dict format from the in-memory dict. The text att is a single str
    def get_document_from_dict_no_paragraph_list(self, cord_uid):
        """
        Given a cord_uid returns the document with the matching id. This method reads the document from the internal
        documents dictionary where all documents have to be previously stored. The 'text' attribute of the document
        is a single string instead of a list of paragraphs
        :param cord_uid: The id of the document to retrieve
        :return: a dictionary containing the document attributes including cord_uid, title and text
        """

        if cord_uid not in self.paper_dict:
            raise Exception("Provided cord_uid does not match any document in our dataset")

        pre_doc = self.paper_dict[cord_uid]
        doc = {'cord_uid': cord_uid, 'title': pre_doc['title'], 'text': ''.join(pre_doc['text'])}
        return doc